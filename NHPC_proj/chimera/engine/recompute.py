"""
Activation checkpointing utilities for Chimera.
Reduces memory by recomputing activations during backward pass.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Callable, Any, Tuple
import logging


logger = logging.getLogger(__name__)


class ActivationCheckpointing:
    """
    Activation checkpointing for Transformer blocks.
    
    Trade-off: ~33% extra backward time for significant memory savings.
    """
    
    @staticmethod
    def checkpoint_function(function: Callable, *args, **kwargs) -> Any:
        """
        Checkpoint wrapper using PyTorch's checkpoint API.
        
        Discards intermediate activations during forward,
        recomputes them during backward.
        
        Args:
            function: Function to checkpoint (typically nn.Module.forward)
            *args: Positional arguments to function
            **kwargs: Keyword arguments to function
        
        Returns:
            Output of function
        """
        # Use PyTorch's checkpoint which handles recompute automatically
        return checkpoint(function, *args, **kwargs, use_reentrant=False)
    
    @staticmethod
    def wrap_attention_block(attention_module: nn.Module) -> nn.Module:
        """
        Wrap self-attention module for selective checkpointing.
        
        Args:
            attention_module: Self-attention nn.Module
        
        Returns:
            Wrapped module with checkpointed forward
        """
        original_forward = attention_module.forward
        
        def checkpointed_forward(*args, **kwargs):
            return ActivationCheckpointing.checkpoint_function(
                original_forward,
                *args,
                **kwargs
            )
        
        attention_module.forward = checkpointed_forward
        return attention_module
    
    @staticmethod
    def wrap_mlp_block(mlp_module: nn.Module) -> nn.Module:
        """
        Wrap MLP/FFN module for selective checkpointing.
        
        Args:
            mlp_module: MLP nn.Module
        
        Returns:
            Wrapped module with checkpointed forward
        """
        original_forward = mlp_module.forward
        
        def checkpointed_forward(*args, **kwargs):
            return ActivationCheckpointing.checkpoint_function(
                original_forward,
                *args,
                **kwargs
            )
        
        mlp_module.forward = checkpointed_forward
        return mlp_module
    
    @staticmethod
    def wrap_transformer_block(
        transformer_block: nn.Module,
        checkpoint_attention: bool = True,
        checkpoint_mlp: bool = True
    ) -> nn.Module:
        """
        Wrap entire Transformer block with selective checkpointing.
        
        Args:
            transformer_block: Transformer block nn.Module
            checkpoint_attention: Whether to checkpoint attention
            checkpoint_mlp: Whether to checkpoint MLP
        
        Returns:
            Wrapped module
        """
        # Assume standard structure: block.attention and block.mlp
        if checkpoint_attention and hasattr(transformer_block, 'attention'):
            transformer_block.attention = ActivationCheckpointing.wrap_attention_block(
                transformer_block.attention
            )
        
        if checkpoint_mlp and hasattr(transformer_block, 'mlp'):
            transformer_block.mlp = ActivationCheckpointing.wrap_mlp_block(
                transformer_block.mlp
            )
        
        return transformer_block
    
    @staticmethod
    def apply_selective_checkpointing(
        model: nn.Module,
        checkpoint_ratio: float = 0.5
    ) -> nn.Module:
        """
        Apply selective checkpointing to a fraction of layers.
        
        Strategy: Checkpoint every other layer to balance memory/compute.
        
        Args:
            model: Model with sequential blocks
            checkpoint_ratio: Fraction of blocks to checkpoint (0.0 to 1.0)
        
        Returns:
            Model with checkpointing applied
        """
        if not hasattr(model, 'blocks'):
            logger.warning("Model has no 'blocks' attribute, skipping checkpointing")
            return model
        
        num_blocks = len(model.blocks)
        checkpoint_every = int(1.0 / checkpoint_ratio) if checkpoint_ratio > 0 else num_blocks + 1
        
        for i, block in enumerate(model.blocks):
            if i % checkpoint_every == 0:
                model.blocks[i] = ActivationCheckpointing.wrap_transformer_block(block)
                logger.debug(f"Applied checkpointing to block {i}")
        
        return model
    
    @staticmethod
    def estimate_recompute_overhead(num_checkpointed_blocks: int) -> float:
        """
        Estimate computational overhead from recomputation.
        
        From paper: Recompute adds ~1/3 to backward time.
        
        Args:
            num_checkpointed_blocks: Number of blocks with checkpointing
        
        Returns:
            Overhead multiplier (e.g., 1.33 = 33% overhead)
        """
        # Empirical: ~1/3 extra backward time per checkpointed block
        overhead_per_block = 0.33
        
        # Aggregate overhead (assumes blocks contribute equally)
        total_overhead = 1.0 + (overhead_per_block * num_checkpointed_blocks)
        
        return total_overhead
    
    @staticmethod
    def estimate_memory_savings(
        activation_memory_mb: float,
        checkpoint_ratio: float
    ) -> float:
        """
        Estimate memory savings from checkpointing.
        
        Args:
            activation_memory_mb: Original activation memory
            checkpoint_ratio: Fraction of activations checkpointed
        
        Returns:
            Saved memory in MB
        """
        # Checkpointed activations don't need to be stored
        # Only minimal state (inputs) kept
        savings_ratio = checkpoint_ratio * 0.8  # ~80% of activation saved
        
        saved_memory = activation_memory_mb * savings_ratio
        
        return saved_memory


class CheckpointedSequential(nn.Sequential):
    """
    Sequential module with automatic checkpointing for each sub-module.
    Alternative to manual wrapping.
    """
    
    def __init__(self, *args, checkpoint_every: int = 1):
        """
        Args:
            *args: Sub-modules to include
            checkpoint_every: Checkpoint every N-th module
        """
        super().__init__(*args)
        self.checkpoint_every = checkpoint_every
    
    def forward(self, x):
        """Forward with selective checkpointing"""
        for i, module in enumerate(self):
            if i % self.checkpoint_every == 0:
                # Checkpoint this module
                x = checkpoint(module, x, use_reentrant=False)
            else:
                # Regular forward
                x = module(x)
        
        return x


class GradientCheckpointWrapper(nn.Module):
    """
    Wrapper module that applies gradient checkpointing to wrapped module.
    """
    
    def __init__(self, module: nn.Module):
        """
        Args:
            module: Module to wrap with checkpointing
        """
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        """Forward with checkpointing"""
        return checkpoint(
            self.module,
            *args,
            **kwargs,
            use_reentrant=False
        )


# Utility functions for integration

def configure_checkpointing_for_forward_doubling(
    model_stage: nn.Module
) -> nn.Module:
    """
    Configure checkpointing for forward doubling mode.
    
    Forward doubling requires aggressive checkpointing due to 2x activations.
    
    Args:
        model_stage: Pipeline stage module
    
    Returns:
        Stage with checkpointing configured
    """
    logger.info("Configuring aggressive checkpointing for forward doubling")
    
    # Checkpoint all blocks for forward doubling
    model_stage = ActivationCheckpointing.apply_selective_checkpointing(
        model_stage,
        checkpoint_ratio=1.0  # Checkpoint everything
    )
    
    return model_stage


def configure_checkpointing_for_memory_budget(
    model_stage: nn.Module,
    current_memory_mb: float,
    memory_budget_mb: float
) -> Tuple[nn.Module, float]:
    """
    Configure checkpointing to fit memory budget.
    
    Args:
        model_stage: Pipeline stage module
        current_memory_mb: Current memory usage
        memory_budget_mb: Target memory budget
    
    Returns:
        Tuple of (configured_stage, estimated_checkpoint_ratio)
    """
    if current_memory_mb <= memory_budget_mb:
        logger.info("Current memory within budget, no checkpointing needed")
        return model_stage, 0.0
    
    # Calculate required memory reduction
    memory_excess = current_memory_mb - memory_budget_mb
    reduction_ratio = memory_excess / current_memory_mb
    
    # Conservative: Need to checkpoint more than reduction ratio
    checkpoint_ratio = min(1.0, reduction_ratio * 1.5)
    
    logger.info(f"Applying {checkpoint_ratio:.2%} checkpointing to reduce memory")
    
    model_stage = ActivationCheckpointing.apply_selective_checkpointing(
        model_stage,
        checkpoint_ratio=checkpoint_ratio
    )
    
    return model_stage, checkpoint_ratio


# Example usage
if __name__ == "__main__":
    # Example: Wrap a simple transformer block
    class SimpleTransformerBlock(nn.Module):
        def __init__(self, hidden_size=128):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )
            self.ln1 = nn.LayerNorm(hidden_size)
            self.ln2 = nn.LayerNorm(hidden_size)
        
        def forward(self, x):
            # Attention
            attn_out, _ = self.attention(x, x, x)
            x = self.ln1(x + attn_out)
            
            # MLP
            mlp_out = self.mlp(x)
            x = self.ln2(x + mlp_out)
            
            return x
    
    # Create and wrap block
    block = SimpleTransformerBlock()
    wrapped_block = ActivationCheckpointing.wrap_transformer_block(block)
    
    # Test forward pass
    x = torch.randn(2, 10, 128)  # (batch, seq_len, hidden)
    output = wrapped_block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Checkpointing applied successfully!")
    
    # Estimate overhead
    overhead = ActivationCheckpointing.estimate_recompute_overhead(1)
    print(f"Estimated recompute overhead: {overhead:.2f}x")
    
    # Estimate memory savings
    savings = ActivationCheckpointing.estimate_memory_savings(100.0, checkpoint_ratio=0.5)
    print(f"Estimated memory savings: {savings:.2f} MB")
