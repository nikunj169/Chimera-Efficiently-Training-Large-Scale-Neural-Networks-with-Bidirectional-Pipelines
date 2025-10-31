"""
Gradient synchronization with eager-sync optimization.
Edge stages use eager allreduce, middle stages defer to post-iteration.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


class AllReduceHandler:
    """
    Manages gradient synchronization across pipeline replicas.
    Implements eager-sync-opt strategy from Chimera paper.
    """
    
    def __init__(self, rank: int, process_groups, eager_sync_stages: set):
        """
        Args:
            rank: Current process rank
            process_groups: ProcessGroups instance
            eager_sync_stages: Set of stage IDs eligible for eager sync
        """
        self.rank = rank
        self.pg = process_groups
        self.eager_sync_stages = eager_sync_stages
        
        # Current stage info
        self.replica_id = self.pg.get_replica_id(rank)
        self.stage_id = self.pg.get_stage_id(rank)
        
        # Gradient buckets by parameter
        self.gradient_buckets: Dict[str, torch.Tensor] = {}
        
        # Pending allreduce operations
        self.pending_allreduces: List[dist.Work] = []
        
        # Eager sync enabled for this stage?
        self.is_eager_stage = self.stage_id in eager_sync_stages
        
        logger.info(f"Rank {self.rank} (stage {self.stage_id}): "
                   f"Eager sync {'ENABLED' if self.is_eager_stage else 'DISABLED'}")
    
    def register_gradients(self, model: torch.nn.Module):
        """
        Register model parameters for gradient synchronization.
        
        Args:
            model: Model stage with parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.gradient_buckets[name] = None
        
        logger.info(f"Rank {self.rank}: Registered {len(self.gradient_buckets)} parameters")
    
    def eager_sync_gradient(self, param_name: str, gradient: torch.Tensor):
        """
        Eagerly synchronize gradient if this is an edge stage.
        Called immediately after backward for edge stages.
        
        Args:
            param_name: Parameter identifier
            gradient: Gradient tensor to synchronize
        """
        if not self.is_eager_stage:
            # Middle stage: defer to post-iteration
            self.gradient_buckets[param_name] = gradient
            return
        
        # Edge stage: launch async allreduce immediately
        stage_group = self.pg.get_stage_group(self.stage_id)
        
        # Clone gradient to avoid modification during communication
        grad_copy = gradient.clone()
        
        # Launch nonblocking allreduce
        work = dist.all_reduce(
            grad_copy,
            op=dist.ReduceOp.SUM,
            group=stage_group,
            async_op=True
        )
        
        self.pending_allreduces.append(work)
        self.gradient_buckets[param_name] = grad_copy
        
        logger.debug(f"Rank {self.rank}: Eager sync launched for {param_name}")
    
    def sync_all_gradients(self, model: torch.nn.Module):
        """
        Synchronize all gradients at iteration end.
        
        For eager stages: Wait for pending operations
        For middle stages: Launch allreduce now
        
        Args:
            model: Model with gradients
        """
        stage_group = self.pg.get_stage_group(self.stage_id)
        
        if self.is_eager_stage:
            # Wait for all pending eager allreduces
            for work in self.pending_allreduces:
                work.wait()
            
            # Copy synchronized gradients back to parameters
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.gradient_buckets:
                    synced_grad = self.gradient_buckets[name]
                    if synced_grad is not None:
                        # Average across replicas
                        param.grad.copy_(synced_grad / self.pg.W)
            
            # Clear pending operations
            self.pending_allreduces.clear()
            
            logger.debug(f"Rank {self.rank}: Eager sync completed")
        
        else:
            # Middle stage: synchronize now (blocking)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    dist.all_reduce(
                        param.grad,
                        op=dist.ReduceOp.SUM,
                        group=stage_group,
                        async_op=False
                    )
                    
                    # Average across replicas
                    param.grad.div_(self.pg.W)
            
            logger.debug(f"Rank {self.rank}: Post-iteration sync completed")
        
        # Clear gradient buckets
        self.gradient_buckets = {k: None for k in self.gradient_buckets}
    
    def compute_allreduce_cost(self, L: int, alpha: float, beta: float) -> float:
        """
        Compute allreduce cost using Rabenseifner's algorithm.
        
        Cost = 2 * log2(W) * alpha + 2 * (W-1) / W * beta * L
        
        Args:
            L: Message size (number of elements)
            alpha: Network latency (seconds)
            beta: Inverse bandwidth (seconds per byte)
        
        Returns:
            Estimated time in seconds
        """
        import math
        
        W = self.pg.W
        
        # Rabenseifner cost model
        latency_cost = 2 * math.log2(W) * alpha
        bandwidth_cost = 2 * (W - 1) / W * beta * L
        
        total_cost = latency_cost + bandwidth_cost
        
        return total_cost
    
    def estimate_overlap_capability(self, schedule) -> float:
        """
        Estimate how much allreduce can overlap with computation bubbles.
        
        Returns:
            Overlap fraction (0.0 to 1.0)
        """
        # Simplified: edge stages have more overlap opportunity
        if self.is_eager_stage:
            # Can overlap with bubbles during forward/backward
            return 0.7  # 70% overlap
        else:
            # Must wait until iteration end
            return 0.0  # No overlap
    
    def barrier(self):
        """Global barrier across all processes"""
        dist.barrier()


# Example usage
if __name__ == "__main__":
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    # Initialize distributed
    dist.init_process_group(backend='gloo', rank=0, world_size=4)
    
    from chimera.dist.groups import init_process_groups
    
    # Create process groups: W=2, D=2
    pg = init_process_groups(world_size=4, W=2, D=2)
    
    # Edge stages: {0, 1} (for D=2)
    eager_stages = {0, 1}
    
    # Create allreduce handler
    allreduce = AllReduceHandler(rank=0, process_groups=pg, eager_sync_stages=eager_stages)
    
    print(f"Rank {allreduce.rank}, Stage {allreduce.stage_id}")
    print(f"Eager sync: {allreduce.is_eager_stage}")
    
    # Test allreduce cost computation
    cost = allreduce.compute_allreduce_cost(L=1000000, alpha=1e-5, beta=1e-9)
    print(f"Estimated allreduce cost: {cost:.6f} seconds")
    
    dist.destroy_process_group()
