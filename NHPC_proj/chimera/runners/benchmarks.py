"""
Microbenchmarking utilities to estimate F_t, B_t, alpha, beta.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import time
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


class Benchmarks:
    """
    Microbenchmarking for performance model parameters.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device to run benchmarks on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Benchmarks initialized on device: {self.device}")
    
    def benchmark_forward_time(
        self,
        model_stage: nn.Module,
        batch_size: int,
        seq_length: int,
        num_iterations: int = 100
    ) -> float:
        """
        Benchmark forward pass time (F_t).
        
        Args:
            model_stage: Model stage to benchmark
            batch_size: Micro-batch size
            seq_length: Sequence length
            num_iterations: Number of iterations to average
        
        Returns:
            Average forward time in seconds
        """
        model_stage = model_stage.to(self.device)
        model_stage.eval()
        
        # Create dummy input
        if hasattr(model_stage, 'embeddings'):
            # Stage 0: input_ids
            dummy_input = torch.randint(0, 30522, (batch_size, seq_length), device=self.device)
        else:
            # Other stages: hidden states
            hidden_size = model_stage.config.hidden_size if hasattr(model_stage, 'config') else 1024
            dummy_input = torch.randn(batch_size, seq_length, hidden_size, device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_stage(dummy_input)
        
        # Benchmark
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model_stage(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_iterations
        
        logger.info(f"Forward time (F_t): {avg_time:.6f}s")
        return avg_time
    
    def benchmark_backward_time(
        self,
        model_stage: nn.Module,
        batch_size: int,
        seq_length: int,
        num_iterations: int = 100
    ) -> float:
        """
        Benchmark backward pass time (B_t).
        
        Args:
            model_stage: Model stage to benchmark
            batch_size: Micro-batch size
            seq_length: Sequence length
            num_iterations: Number of iterations to average
        
        Returns:
            Average backward time in seconds
        """
        model_stage = model_stage.to(self.device)
        model_stage.train()
        
        # Create dummy input
        if hasattr(model_stage, 'embeddings'):
            dummy_input = torch.randint(0, 30522, (batch_size, seq_length), device=self.device)
        else:
            hidden_size = model_stage.config.hidden_size if hasattr(model_stage, 'config') else 1024
            dummy_input = torch.randn(batch_size, seq_length, hidden_size, device=self.device, requires_grad=True)
        
        # Warmup
        for _ in range(10):
            output = model_stage(dummy_input)
            loss = output.sum()
            loss.backward()
            model_stage.zero_grad()
        
        # Benchmark
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        for _ in range(num_iterations):
            output = model_stage(dummy_input)
            loss = output.sum()
            loss.backward()
            model_stage.zero_grad()
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / num_iterations
        
        logger.info(f"Backward time (B_t): {avg_time:.6f}s")
        return avg_time
    
    def benchmark_network_params(
        self,
        message_sizes: list = None,
        num_iterations: int = 100
    ) -> Tuple[float, float]:
        """
        Benchmark network parameters alpha (latency) and beta (inverse bandwidth).
        
        Requires distributed setup.
        
        Args:
            message_sizes: List of message sizes to test (bytes)
            num_iterations: Number of iterations per size
        
        Returns:
            (alpha, beta) tuple
        """
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, using default values")
            return 1e-5, 1e-9  # Default: 10us latency, 1GB/s bandwidth
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        if world_size < 2:
            logger.warning("Need at least 2 processes for network benchmarking")
            return 1e-5, 1e-9
        
        if message_sizes is None:
            message_sizes = [1024, 10240, 102400, 1024000, 10240000]  # 1KB to 10MB
        
        results = []
        
        for size in message_sizes:
            # Create tensors
            tensor = torch.randn(size // 4, device=self.device)  # fp32 = 4 bytes
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            dist.barrier()
            start_time = time.perf_counter()
            
            for _ in range(num_iterations):
                if rank == 0:
                    dist.send(tensor, dst=1)
                elif rank == 1:
                    dist.recv(tensor, src=0)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            dist.barrier()
            end_time = time.perf_counter()
            
            avg_time = (end_time - start_time) / num_iterations
            results.append((size, avg_time))
        
        # Linear regression: time = alpha + beta * size
        # y = a + b*x
        n = len(results)
        sum_x = sum(size for size, _ in results)
        sum_y = sum(time for _, time in results)
        sum_xy = sum(size * time for size, time in results)
        sum_x2 = sum(size ** 2 for size, _ in results)
        
        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        alpha = (sum_y - beta * sum_x) / n
        
        logger.info(f"Network parameters: alpha={alpha:.6f}s, beta={beta:.9f}s/byte")
        return alpha, beta
    
    def run_all_benchmarks(
        self,
        model_stage: nn.Module,
        batch_size: int,
        seq_length: int
    ) -> Dict[str, float]:
        """
        Run all benchmarks and return results.
        
        Args:
            model_stage: Model stage to benchmark
            batch_size: Micro-batch size
            seq_length: Sequence length
        
        Returns:
            Dict with all benchmark results
        """
        results = {}
        
        # Compute benchmarks
        results['F_t'] = self.benchmark_forward_time(model_stage, batch_size, seq_length)
        results['B_t'] = self.benchmark_backward_time(model_stage, batch_size, seq_length)
        
        # Network benchmarks (if distributed)
        if dist.is_initialized():
            alpha, beta = self.benchmark_network_params()
            results['alpha'] = alpha
            results['beta'] = beta
        else:
            results['alpha'] = 1e-5
            results['beta'] = 1e-9
        
        return results


# Example usage
if __name__ == "__main__":
    from chimera.models.bert48 import BertConfig, BertStage
    
    # Create a BERT stage
    config = BertConfig(num_hidden_layers=12)
    stage = BertStage(config, stage_id=0, layer_range=(0, 12))
    
    # Run benchmarks
    benchmarker = Benchmarks(device='cpu')
    
    results = benchmarker.run_all_benchmarks(
        model_stage=stage,
        batch_size=4,
        seq_length=128
    )
    
    print("\nBenchmark Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.6f}")
