"""
Pipeline stage runtime for Chimera - FULLY INTEGRATED with Person B's dist handlers.
Executes scheduled forward/backward operations with activation management.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

from .schedule import ScheduleSlot, BidirectionalSchedule
from .recompute import ActivationCheckpointing

# Import Person B's handlers
from chimera.dist import P2PHandler, AllReduceHandler


logger = logging.getLogger(__name__)


@dataclass
class MicroBatchContext:
    """Context for a single micro-batch during execution"""
    micro_batch_id: int
    input_tensor: Optional[torch.Tensor] = None
    output_tensor: Optional[torch.Tensor] = None
    input_grad: Optional[torch.Tensor] = None
    checkpointed: bool = False
    rng_state: Optional[Any] = None


class StageWorker:
    """
    Pipeline stage worker executing scheduled operations.
    FULLY INTEGRATED with Person B's P2P and AllReduce handlers.
    """
    
    def __init__(
        self,
        rank: int,
        stage_id: int,
        model_stage: nn.Module,
        schedule: List[ScheduleSlot],
        p2p_handler: P2PHandler,  # Person B's P2P handler
        allreduce_handler: AllReduceHandler,  # Person B's AllReduce handler
        loss_fn: Optional[nn.Module] = None,
        enable_recompute: bool = False,
        memory_budget_mb: Optional[float] = None
    ):
        """
        Args:
            rank: Worker rank in distributed group
            stage_id: Logical stage ID (0 to D-1)
            model_stage: nn.Module containing layers for this stage
            schedule: List of ScheduleSlots to execute
            p2p_handler: Person B's P2PHandler instance
            allreduce_handler: Person B's AllReduceHandler instance
            loss_fn: Loss function (only for final stage)
            enable_recompute: Enable activation checkpointing
            memory_budget_mb: Memory budget for activation management
        """
        self.rank = rank
        self.stage_id = stage_id
        self.model_stage = model_stage
        self.schedule = schedule
        
        # ✅ INTEGRATION: Use Person B's handlers
        self.p2p = p2p_handler
        self.allreduce = allreduce_handler
        
        self.loss_fn = loss_fn
        
        # Activation management
        self.activation_stash: Dict[int, MicroBatchContext] = {}
        self.enable_recompute = enable_recompute
        self.memory_budget_mb = memory_budget_mb
        self.checkpointer = ActivationCheckpointing() if enable_recompute else None
        
        # Statistics
        self.forward_count = 0
        self.backward_count = 0
        self.recompute_count = 0
        
        # Data loader (for stage 0)
        self.data_iterator = None
        self.batch_cache: Dict[int, Any] = {}
        
        # Determine number of stages from schedule
        self.num_stages = self._infer_num_stages()
        
        logger.info(f"StageWorker initialized: rank={rank}, stage={stage_id}, "
                   f"num_stages={self.num_stages}, recompute={enable_recompute}")
        
    def set_data_iterator(self, data_iterator):
        """Set data loader iterator for stage 0"""
        self.data_iterator = data_iterator
        
    def run_iteration(self) -> Dict[str, float]:
        """
        Execute one training iteration (all scheduled operations).
        FULLY INTEGRATED with Person B's handlers.
        
        Returns:
            Dict with iteration statistics
        """
        logger.info(f"Rank {self.rank} starting iteration with {len(self.schedule)} operations")
        
        # Reset counters
        self.forward_count = 0
        self.backward_count = 0
        self.recompute_count = 0
        
        # Execute all scheduled slots in order
        for slot in self.schedule:
            if slot.operation == 'forward':
                self._forward_pass(slot)
            elif slot.operation == 'backward':
                self._backward_pass(slot)
            else:
                raise ValueError(f"Unknown operation: {slot.operation}")
        
        # ✅ INTEGRATION: Synchronize gradients using Person B's AllReduce
        self.allreduce.sync_all_gradients(self.model_stage)
        
        # Iteration flush
        stats = self._flush_iteration()
        
        return stats
    
    def _forward_pass(self, slot: ScheduleSlot):
        """
        Execute forward pass for a micro-batch.
        ✅ INTEGRATION: Uses Person B's P2P for communication.
        """
        mb_id = slot.micro_batch_id
        stage_id = slot.stage_id
        
        logger.debug(f"Rank {self.rank} forward mb={mb_id} at time {slot.time}")
        
        # Step 1: Get input tensor
        if stage_id == 0:
            # First stage: Load from data iterator
            input_tensor = self._get_batch_from_loader(mb_id)
        else:
            # ✅ INTEGRATION: Receive from previous stage via Person B's P2P
            input_tensor = self.p2p.recv_activation(
                src_stage=stage_id - 1,
                micro_batch_id=mb_id,
                blocking=True
            )
        
        # Ensure input requires gradient for autograd
        if not input_tensor.requires_grad and stage_id > 0:
            input_tensor.requires_grad = True
        
        # Step 2: Forward computation
        if slot.doubled_forward or self.enable_recompute:
            output_tensor = self._forward_with_checkpoint(input_tensor)
        else:
            output_tensor = self.model_stage(input_tensor)
        
        # Step 3: Stash activations
        context = MicroBatchContext(
            micro_batch_id=mb_id,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            checkpointed=slot.doubled_forward or self.enable_recompute
        )
        
        if context.checkpointed:
            context.rng_state = torch.get_rng_state()
            
        self.activation_stash[mb_id] = context
        
        # Memory management
        if self.memory_budget_mb:
            self._manage_activation_memory()
        
        # Step 4: Send to next stage
        if stage_id < self.num_stages - 1:
            # ✅ INTEGRATION: Send via Person B's P2P
            self.p2p.send_activation(
                tensor=output_tensor.detach(),
                dst_stage=stage_id + 1,
                micro_batch_id=mb_id,
                blocking=True
            )
        
        self.forward_count += 1
    
    def _backward_pass(self, slot: ScheduleSlot):
        """
        Execute backward pass for a micro-batch.
        ✅ INTEGRATION: Uses Person B's P2P and eager gradient sync.
        """
        mb_id = slot.micro_batch_id
        stage_id = slot.stage_id
        
        logger.debug(f"Rank {self.rank} backward mb={mb_id} at time {slot.time}")
        
        # Step 1: Get output gradient
        if stage_id == self.num_stages - 1:
            # Final stage: Compute loss gradient
            output_grad = self._compute_loss_gradient(mb_id)
        else:
            # ✅ INTEGRATION: Receive gradient from next stage via Person B's P2P
            output_grad = self.p2p.recv_gradient(
                src_stage=stage_id + 1,
                micro_batch_id=mb_id,
                blocking=True
            )
        
        # Step 2: Retrieve or recompute activations
        if mb_id in self.activation_stash:
            context = self.activation_stash[mb_id]
            
            if slot.requires_recompute and context.checkpointed:
                context = self._recompute_activations(mb_id, context)
                self.recompute_count += 1
        else:
            logger.warning(f"Activation for mb {mb_id} not in stash, recomputing...")
            context = self._recompute_activations(mb_id, None)
            self.recompute_count += 1
        
        # Step 3: Backward computation
        output_tensor = context.output_tensor
        
        if output_tensor.requires_grad:
            output_tensor.backward(output_grad)
        else:
            output_tensor.requires_grad = True
            output_tensor.backward(output_grad)
        
        # Get input gradient
        input_grad = context.input_tensor.grad
        
        # Step 4: Send input gradient to previous stage
        if stage_id > 0:
            # ✅ INTEGRATION: Send gradient via Person B's P2P
            self.p2p.send_gradient(
                tensor=input_grad.detach(),
                dst_stage=stage_id - 1,
                micro_batch_id=mb_id,
                blocking=True
            )
        
        # ✅ INTEGRATION: Eager gradient sync for edge stages
        # This is where Person B's eager-sync-opt happens
        for name, param in self.model_stage.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.allreduce.eager_sync_gradient(name, param.grad)
        
        # Step 5: Free activation memory
        self._free_activations(mb_id)
        
        self.backward_count += 1
    
    def _forward_with_checkpoint(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation checkpointing"""
        if self.checkpointer:
            return self.checkpointer.checkpoint_function(
                self.model_stage,
                input_tensor
            )
        else:
            return self.model_stage(input_tensor)
    
    def _get_batch_from_loader(self, mb_id: int) -> torch.Tensor:
        """Load batch from data iterator (stage 0 only)"""
        if mb_id in self.batch_cache:
            return self.batch_cache[mb_id]
        
        if self.data_iterator is None:
            # Mock data for testing
            return torch.randint(0, 30522, (4, 128))
        
        try:
            batch = next(self.data_iterator)
            
            if isinstance(batch, (tuple, list)):
                input_tensor = batch[0]
            elif isinstance(batch, dict):
                input_tensor = batch['input_ids']
            else:
                input_tensor = batch
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            self.batch_cache[mb_id] = input_tensor
            return input_tensor
            
        except StopIteration:
            raise RuntimeError("Data iterator exhausted")
    
    def _compute_loss_gradient(self, mb_id: int) -> torch.Tensor:
        """Compute loss and return gradient (final stage only)"""
        if mb_id not in self.activation_stash:
            raise RuntimeError(f"No activation for mb {mb_id} to compute loss")
        
        context = self.activation_stash[mb_id]
        output_tensor = context.output_tensor
        
        # Mock loss for testing (in real training, use actual labels)
        if self.loss_fn is None:
            loss = output_tensor.sum()
        else:
            target = self._get_target_for_batch(mb_id)
            loss = self.loss_fn(output_tensor, target)
        
        loss.backward()
        return output_tensor.grad
    
    def _get_target_for_batch(self, mb_id: int) -> torch.Tensor:
        """Get target labels for loss computation"""
        # Mock implementation - override in real training
        return torch.zeros(4, 128, dtype=torch.long)
    
    def _recompute_activations(
        self,
        mb_id: int,
        old_context: Optional[MicroBatchContext]
    ) -> MicroBatchContext:
        """Recompute forward pass for a micro-batch"""
        logger.debug(f"Recomputing activations for mb {mb_id}")
        
        if old_context and old_context.rng_state:
            torch.set_rng_state(old_context.rng_state)
        
        if old_context and old_context.input_tensor is not None:
            input_tensor = old_context.input_tensor
        else:
            if self.stage_id == 0:
                input_tensor = self._get_batch_from_loader(mb_id)
            else:
                raise RuntimeError(f"Cannot recompute mb {mb_id} without input")
        
        input_tensor.requires_grad = True
        
        with torch.enable_grad():
            output_tensor = self.model_stage(input_tensor)
        
        context = MicroBatchContext(
            micro_batch_id=mb_id,
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            checkpointed=False
        )
        
        return context
    
    def _free_activations(self, mb_id: int):
        """Free activation memory for a micro-batch"""
        if mb_id in self.activation_stash:
            del self.activation_stash[mb_id]
            
        if mb_id in self.batch_cache:
            del self.batch_cache[mb_id]
    
    def _manage_activation_memory(self):
        """Evict activations if memory budget exceeded"""
        if not self.memory_budget_mb:
            return
        
        current_memory = self._estimate_stash_memory()
        
        if current_memory > self.memory_budget_mb:
            sorted_mbs = sorted(self.activation_stash.keys())
            
            for mb_id in sorted_mbs:
                if current_memory <= self.memory_budget_mb:
                    break
                
                context = self.activation_stash[mb_id]
                
                if context.checkpointed or self.stage_id == 0:
                    if context.output_tensor is not None:
                        del context.output_tensor
                        context.output_tensor = None
                    
                    current_memory = self._estimate_stash_memory()
    
    def _estimate_stash_memory(self) -> float:
        """Estimate memory usage of activation stash in MB"""
        total_bytes = 0
        
        for context in self.activation_stash.values():
            if context.input_tensor is not None:
                total_bytes += context.input_tensor.nelement() * context.input_tensor.element_size()
            
            if context.output_tensor is not None:
                total_bytes += context.output_tensor.nelement() * context.output_tensor.element_size()
        
        return total_bytes / (1024 ** 2)
    
    def _flush_iteration(self) -> Dict[str, float]:
        """Flush iteration: Synchronous completion signal"""
        logger.info(f"Rank {self.rank} flushing iteration: "
                   f"{self.forward_count} forwards, {self.backward_count} backwards, "
                   f"{self.recompute_count} recomputes")
        
        # Clear all caches
        self.activation_stash.clear()
        self.batch_cache.clear()
        
        stats = {
            'forward_count': self.forward_count,
            'backward_count': self.backward_count,
            'recompute_count': self.recompute_count,
            'peak_memory_mb': self._estimate_stash_memory()
        }
        
        return stats
    
    def _infer_num_stages(self) -> int:
        """Get total number of pipeline stages from schedule"""
        if not self.schedule:
            return 1
        max_stage = max(slot.stage_id for slot in self.schedule)
        return max_stage + 1


# Example usage showing full integration
if __name__ == "__main__":
    import torch.distributed as dist
    from chimera.dist import init_process_groups, P2PHandler, AllReduceHandler
    from chimera.engine import BidirectionalSchedule
    from chimera.models.bert48 import BertConfig, BertStage
    
    # Mock distributed setup
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    
    dist.init_process_group(backend='gloo', rank=0, world_size=4)
    
    # ✅ INTEGRATION EXAMPLE
    # 1. Initialize Person B's process groups
    pg = init_process_groups(world_size=4, W=2, D=2)
    
    # 2. Create Person B's handlers
    p2p = P2PHandler(rank=0, process_groups=pg)
    allreduce = AllReduceHandler(rank=0, process_groups=pg, eager_sync_stages={0, 1})
    
    # 3. Create Person A's schedule
    scheduler = BidirectionalSchedule(D=2, N=2)
    schedule = scheduler.build_schedule()
    rank_schedule = schedule[0]
    
    # 4. Create Person A's model
    config = BertConfig(num_hidden_layers=4, hidden_size=128)
    stage = BertStage(config, stage_id=0, layer_range=(0, 2))
    
    # 5. Create FULLY INTEGRATED StageWorker
    worker = StageWorker(
        rank=0,
        stage_id=0,
        model_stage=stage,
        schedule=rank_schedule,
        p2p_handler=p2p,  # ✅ Person B's handler
        allreduce_handler=allreduce,  # ✅ Person B's handler
        enable_recompute=False
    )
    
    print(f"✅ Fully integrated StageWorker created!")
    print(f"   Using P2P handler: {type(p2p).__name__}")
    print(f"   Using AllReduce handler: {type(allreduce).__name__}")
    
    dist.destroy_process_group()
