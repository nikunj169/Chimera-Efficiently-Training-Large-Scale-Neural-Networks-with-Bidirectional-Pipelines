"""
Performance model for Chimera pipeline parallelism.
Implements Equation 1 from the paper.
"""

import math
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


class PerformanceModel:
    """
    Performance model: T = (F_t + Comm_p2p) * C_f + (B_t + Comm_p2p) * C_b + max UnoverlappedAllreduce
    
    Where:
    - F_t: Forward time per micro-batch
    - B_t: Backward time per micro-batch
    - Comm_p2p: Point-to-point communication time
    - C_f, C_b: Critical path counts from schedule
    - UnoverlappedAllreduce: Allreduce cost that cannot overlap
    """
    
    def __init__(
        self,
        alpha: float,
        beta: float,
        F_t: float,
        recompute_enabled: bool = False
    ):
        """
        Args:
            alpha: Network latency (seconds)
            beta: Inverse bandwidth (seconds per byte)
            F_t: Forward time per micro-batch (seconds)
            recompute_enabled: Whether activation recomputation is enabled
        """
        self.alpha = alpha
        self.beta = beta
        self.F_t = F_t
        
        # Backward time: 2x forward (or 3x with recompute)
        if recompute_enabled:
            self.B_t = 3.0 * F_t  # +33% overhead from recompute
        else:
            self.B_t = 2.0 * F_t
        
        logger.info(f"PerformanceModel: F_t={F_t:.6f}s, B_t={self.B_t:.6f}s, "
                   f"alpha={alpha:.6f}s, beta={beta:.9f}s/byte")
    
    def estimate_iteration_time(
        self,
        D: int,
        N: int,
        W: int,
        C_f: int,
        C_b: int,
        message_size_bytes: int,
        eager_sync_stages: set
    ) -> Dict[str, float]:
        """
        Estimate total iteration time.
        
        Args:
            D: Number of pipeline stages
            N: Number of micro-batches
            W: Number of replicas
            C_f: Critical path forward count
            C_b: Critical path backward count
            message_size_bytes: Size of activation/gradient message
            eager_sync_stages: Stages using eager synchronization
        
        Returns:
            Dict with time breakdown
        """
        # P2P communication time per micro-batch
        Comm_p2p = self._compute_p2p_time(message_size_bytes)
        
        # Forward phase time
        forward_time = (self.F_t + Comm_p2p) * C_f
        
        # Backward phase time
        backward_time = (self.B_t + Comm_p2p) * C_b
        
        # Allreduce time (Rabenseifner algorithm)
        allreduce_time = self._compute_allreduce_time(W, message_size_bytes)
        
        # Overlap estimation
        overlap_fraction = self._estimate_overlap(D, eager_sync_stages)
        unoverlapped_allreduce = allreduce_time * (1.0 - overlap_fraction)
        
        # Total time
        total_time = forward_time + backward_time + unoverlapped_allreduce
        
        return {
            'total_time': total_time,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'allreduce_time': allreduce_time,
            'unoverlapped_allreduce': unoverlapped_allreduce,
            'overlap_fraction': overlap_fraction,
            'throughput': N / total_time  # micro-batches per second
        }
    
    def _compute_p2p_time(self, message_size_bytes: int) -> float:
        """
        Compute point-to-point communication time.
        
        Time = alpha + beta * message_size
        """
        return self.alpha + self.beta * message_size_bytes
    
    def _compute_allreduce_time(self, W: int, message_size_bytes: int) -> float:
        """
        Compute allreduce time using Rabenseifner's algorithm.
        
        Time = 2 * log2(W) * alpha + 2 * (W-1)/W * beta * message_size
        """
        if W == 1:
            return 0.0  # No communication needed
        
        latency_term = 2 * math.log2(W) * self.alpha
        bandwidth_term = 2 * (W - 1) / W * self.beta * message_size_bytes
        
        return latency_term + bandwidth_term
    
    def _estimate_overlap(self, D: int, eager_sync_stages: set) -> float:
        """
        Estimate fraction of allreduce that can overlap with computation.
        
        Eager sync on edge stages allows overlap with bubbles.
        """
        num_eager_stages = len(eager_sync_stages)
        total_stages = D
        
        # More eager stages → more overlap
        # Rule of thumb: 50% overlap per eager stage
        overlap = 0.5 * (num_eager_stages / total_stages)
        
        return min(overlap, 0.7)  # Cap at 70%
    
    def compare_configurations(
        self,
        configs: list,
        N: int,
        message_size_bytes: int
    ) -> Tuple[Dict, list]:
        """
        Compare multiple (W, D) configurations.
        
        Args:
            configs: List of (W, D, C_f, C_b) tuples
            N: Number of micro-batches
            message_size_bytes: Message size
        
        Returns:
            (best_config, all_results)
        """
        results = []
        
        for W, D, C_f, C_b in configs:
            eager_stages = {0, D - 1}  # Edge stages
            
            perf = self.estimate_iteration_time(
                D=D,
                N=N,
                W=W,
                C_f=C_f,
                C_b=C_b,
                message_size_bytes=message_size_bytes,
                eager_sync_stages=eager_stages
            )
            
            results.append({
                'W': W,
                'D': D,
                'C_f': C_f,
                'C_b': C_b,
                'performance': perf
            })
        
        # Find best (minimum total time)
        best = min(results, key=lambda x: x['performance']['total_time'])
        
        return best, results


# Example usage
if __name__ == "__main__":
    # Typical values
    alpha = 1e-5  # 10 microseconds latency
    beta = 1e-9   # 1 GB/s bandwidth → 1 ns per byte
    F_t = 0.1     # 100 ms per forward micro-batch
    
    model = PerformanceModel(alpha=alpha, beta=beta, F_t=F_t)
    
    # Estimate for D=4, N=8, W=2
    D, N, W = 4, 8, 2
    C_f = 2 * N + D - 2  # From Chimera schedule
    C_b = 2 * N + D - 2
    message_size = 1024 * 1024 * 4  # 4 MB (fp32)
    
    perf = model.estimate_iteration_time(
        D=D,
        N=N,
        W=W,
        C_f=C_f,
        C_b=C_b,
        message_size_bytes=message_size,
        eager_sync_stages={0, 3}
    )
    
    print("Performance Estimate:")
    print(f"  Total time: {perf['total_time']:.3f}s")
    print(f"  Forward: {perf['forward_time']:.3f}s")
    print(f"  Backward: {perf['backward_time']:.3f}s")
    print(f"  Allreduce (unoverlapped): {perf['unoverlapped_allreduce']:.3f}s")
    print(f"  Throughput: {perf['throughput']:.2f} micro-batches/s")
