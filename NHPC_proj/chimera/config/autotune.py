"""
Automatic configuration selection for Chimera.
Chooses optimal (W, D, B) given total processes and memory constraints.
"""

import math
from typing import Dict, List, Tuple, Optional
import logging

from .perf_model import PerformanceModel


logger = logging.getLogger(__name__)


class AutoTuner:
    """
    Automatically selects optimal pipeline configuration.
    """
    
    def __init__(
        self,
        perf_model: PerformanceModel,
        total_processes: int,
        memory_budget_gb: float,
        model_config: dict
    ):
        """
        Args:
            perf_model: Performance model instance
            total_processes: Total number of processes (P)
            memory_budget_gb: Memory budget per device (GB)
            model_config: Model configuration dict
        """
        self.perf_model = perf_model
        self.P = total_processes
        self.memory_budget_gb = memory_budget_gb
        self.model_config = model_config
        
        logger.info(f"AutoTuner: P={total_processes}, memory={memory_budget_gb}GB")
    
    def select_configuration(
        self,
        target_batch_size: int,
        strategy: str = 'throughput'
    ) -> Dict:
        """
        Select optimal (W, D, B) configuration.
        
        Args:
            target_batch_size: Target global batch size
            strategy: Optimization strategy ('throughput' or 'memory')
        
        Returns:
            Dict with selected configuration
        """
        # Enumerate valid (W, D) pairs
        candidates = self._enumerate_candidates()
        
        if not candidates:
            raise ValueError("No valid (W, D) configurations found")
        
        # For each candidate, find max micro-batch size B
        evaluated = []
        
        for W, D in candidates:
            # Estimate memory per stage
            max_B = self._compute_max_microbatch_size(W, D)
            
            if max_B == 0:
                continue
            
            # Compute N (number of micro-batches per replica)
            N = target_batch_size // (W * max_B)
            
            if N < 2:
                N = 2  # Minimum for pipeline
                max_B = target_batch_size // (W * N)
            
            # Ensure N >= D for good utilization
            if N < D:
                N = D
                max_B = target_batch_size // (W * N)
            
            # Critical path counts from schedule
            C_f = 2 * N + D - 2
            C_b = 2 * N + D - 2
            
            # Message size (approximate)
            hidden_size = self.model_config.get('hidden_size', 1024)
            seq_len = self.model_config.get('max_sequence_length', 512)
            message_size = max_B * seq_len * hidden_size * 4  # fp32
            
            # Evaluate performance
            perf = self.perf_model.estimate_iteration_time(
                D=D,
                N=N,
                W=W,
                C_f=C_f,
                C_b=C_b,
                message_size_bytes=message_size,
                eager_sync_stages={0, D - 1}
            )
            
            evaluated.append({
                'W': W,
                'D': D,
                'B': max_B,
                'N': N,
                'global_batch_size': W * N * max_B,
                'performance': perf
            })
        
        # Select best based on strategy
        if strategy == 'throughput':
            best = max(evaluated, key=lambda x: x['performance']['throughput'])
        else:  # memory
            best = min(evaluated, key=lambda x: x['B'])
        
        # Determine schedule strategy
        schedule_strategy = self._select_schedule_strategy(best['N'], best['D'])
        best['schedule_strategy'] = schedule_strategy
        
        logger.info(f"Selected configuration: W={best['W']}, D={best['D']}, "
                   f"B={best['B']}, N={best['N']}, strategy={schedule_strategy}")
        
        return best
    
    def _enumerate_candidates(self) -> List[Tuple[int, int]]:
        """
        Enumerate valid (W, D) pairs where P = W × D and D is even.
        
        Returns:
            List of (W, D) tuples
        """
        candidates = []
        
        # Find all divisors of P
        for D in range(2, self.P + 1, 2):  # D must be even
            if self.P % D == 0:
                W = self.P // D
                candidates.append((W, D))
        
        logger.debug(f"Candidate configurations: {candidates}")
        return candidates
    
    def _compute_max_microbatch_size(self, W: int, D: int) -> int:
        """
        Compute maximum micro-batch size that fits in memory.
        
        Args:
            W: Number of replicas
            D: Number of stages
        
        Returns:
            Maximum micro-batch size
        """
        from chimera.engine.partition import StagePartitioner
        
        # Create partitioner
        partitioner = StagePartitioner(
            num_stages=D,
            model_config=self.model_config
        )
        
        partitions = partitioner.partition_even_blocks()
        
        # Binary search for max B
        left, right = 1, 1024
        max_valid_B = 0
        
        while left <= right:
            mid = (left + right) // 2
            
            # Check if this B fits in memory
            memory_profile = partitioner.get_memory_profile(partitions, micro_batch_size=mid)
            
            max_memory_gb = max(
                (mem.weight_memory_mb + mem.peak_activation_mb) / 1024
                for mem in memory_profile.values()
            )
            
            if max_memory_gb <= self.memory_budget_gb:
                max_valid_B = mid
                left = mid + 1
            else:
                right = mid - 1
        
        logger.debug(f"Max micro-batch size for W={W}, D={D}: B={max_valid_B}")
        return max_valid_B
    
    def _select_schedule_strategy(self, N: int, D: int) -> str:
        """
        Select schedule strategy based on N and D.
        
        Args:
            N: Number of micro-batches
            D: Number of stages
        
        Returns:
            Strategy name
        """
        if N <= D:
            return 'BASE'
        elif N <= 2 * D:
            return 'DIRECT_CONCAT'
        else:
            # For large N, use forward doubling if memory allows
            return 'FORWARD_DOUBLING'


# Example usage
if __name__ == "__main__":
    from chimera.config.perf_model import PerformanceModel
    
    # Setup performance model
    perf_model = PerformanceModel(
        alpha=1e-5,
        beta=1e-9,
        F_t=0.1,
        recompute_enabled=False
    )
    
    # Model config (BERT-48)
    model_config = {
        'num_layers': 48,
        'hidden_size': 1024,
        'vocab_size': 30522,
        'num_attention_heads': 16,
        'intermediate_size': 4096,
        'max_sequence_length': 512,
        'dtype_bytes': 2
    }
    
    # Create autotuner
    autotuner = AutoTuner(
        perf_model=perf_model,
        total_processes=16,
        memory_budget_gb=16.0,
        model_config=model_config
    )
    
    # Select configuration for batch size 64
    config = autotuner.select_configuration(target_batch_size=64)
    
    print("\nSelected Configuration:")
    print(f"  W (replicas): {config['W']}")
    print(f"  D (stages): {config['D']}")
    print(f"  B (micro-batch): {config['B']}")
    print(f"  N (micro-batches): {config['N']}")
    print(f"  Schedule strategy: {config['schedule_strategy']}")
    print(f"  Throughput: {config['performance']['throughput']:.2f} micro-batches/s")
