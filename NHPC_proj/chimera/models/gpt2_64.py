"""
GPT-2 Large model (64 layers) for Chimera pipeline parallelism.
~1.3B parameter model based on GPT-2 architecture.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class GPT2Config:
    """Configuration for GPT-2 Large (64 layers)"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 1280,
        n_layer: int = 64,
        n_head: int = 20,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu_new",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon


class GPT2Attention(nn.Module):
    """GPT-2 multi-head causal self-attention"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        
        max_positions = config.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            )
        )
        
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}"
            )
        
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
    
    def _split_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def _attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """Compute attention"""
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights * self.scale
        
        # Causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(causal_mask, attn_weights, torch.tensor(mask_value, dtype=attn_weights.dtype))
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_length, n_embd]
        
        Returns:
            attn_output: [batch, seq_length, n_embd]
        """
        # QKV projection
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # Split heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        # Attention
        attn_output = self._attn(query, key, value)
        
        # Merge heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        
        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output


class GPT2MLP(nn.Module):
    """GPT-2 MLP/FFN"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        embed_dim = config.n_embd
        inner_dim = config.n_inner
        
        self.c_fc = nn.Linear(embed_dim, inner_dim)
        self.c_proj = nn.Linear(inner_dim, embed_dim)
        self.dropout = nn.Dropout(config.resid_pdrop)
        
        self.act = self._get_activation(config.activation_function)
    
    def _get_activation(self, activation_function: str):
        """Get activation function"""
        if activation_function == "gelu_new":
            return lambda x: 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        elif activation_function == "gelu":
            return nn.functional.gelu
        else:
            raise ValueError(f"Unknown activation: {activation_function}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    """Single GPT-2 Transformer block"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        hidden_size = config.n_embd
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_length, n_embd]
        
        Returns:
            output: [batch, seq_length, n_embd]
        """
        # Attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)
        hidden_states = residual + attn_output
        
        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states


class GPT2Embeddings(nn.Module):
    """GPT-2 token + position embeddings"""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_length]
        
        Returns:
            embeddings: [batch, seq_length, n_embd]
        """
        batch_size, seq_length = input_ids.size()
        
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        
        return hidden_states


class GPT2Stage(nn.Module):
    """
    Pipeline stage containing subset of GPT-2 layers.
    Stage 0: Embeddings + first K blocks
    Other stages: K blocks each
    """
    
    def __init__(
        self,
        config: GPT2Config,
        stage_id: int,
        layer_range: Tuple[int, int]
    ):
        """
        Args:
            config: GPT-2 configuration
            stage_id: Stage identifier (0 to D-1)
            layer_range: (start_layer, end_layer) for this stage
        """
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.layer_range = layer_range
        
        # Stage 0: Include embeddings
        if stage_id == 0:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.wpe = nn.Embedding(config.n_positions, config.n_embd)
            self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        start_layer, end_layer = layer_range
        self.h = nn.ModuleList([
            GPT2Block(config)
            for _ in range(start_layer, end_layer)
        ])
        
        # Final stage: Include LM head
        self.is_final_stage = (end_layer == config.n_layer)
        if self.is_final_stage:
            self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_data: [batch, seq_length] for stage 0 (input_ids)
                       [batch, seq_length, n_embd] for other stages
        
        Returns:
            hidden_states: [batch, seq_length, n_embd]
        """
        # Stage 0: Embedding layer
        if self.stage_id == 0:
            input_ids = input_data
            batch_size, seq_length = input_ids.size()
            
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            
            hidden_states = inputs_embeds + position_embeds
            hidden_states = self.drop(hidden_states)
        else:
            hidden_states = input_data
        
        # Transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states)
        
        # Final stage: Apply final layer norm
        if self.is_final_stage:
            hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class GPT2ForPipelineParallelism(nn.Module):
    """
    Complete GPT-2 64-layer model partitioned for pipeline parallelism.
    """
    
    def __init__(
        self,
        config: GPT2Config,
        num_stages: int,
        partitions: Optional[list] = None
    ):
        """
        Args:
            config: GPT-2 configuration
            num_stages: Number of pipeline stages
            partitions: List of (start_layer, end_layer) tuples
        """
        super().__init__()
        self.config = config
        self.num_stages = num_stages
        
        # Auto-partition if not provided
        if partitions is None:
            from ..engine.partition import StagePartitioner
            
            partitioner = StagePartitioner(
                num_stages=num_stages,
                model_config={
                    'num_layers': config.n_layer,
                    'hidden_size': config.n_embd,
                    'vocab_size': config.vocab_size,
                    'num_attention_heads': config.n_head,
                    'intermediate_size': config.n_inner,
                    'max_sequence_length': config.n_positions,
                    'dtype_bytes': 2
                }
            )
            partitions = partitioner.partition_even_blocks()
        
        # Create stages
        self.stages = nn.ModuleList([
            GPT2Stage(config, stage_id, layer_range)
            for stage_id, layer_range in enumerate(partitions)
        ])
        
        # LM head (shared with token embeddings)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def get_stage(self, stage_id: int) -> GPT2Stage:
        """Get specific pipeline stage"""
        return self.stages[stage_id]
    
    def get_lm_head(self) -> nn.Linear:
        """Get language modeling head (for final stage)"""
        return self.lm_head


# Example usage
if __name__ == "__main__":
    # Create GPT-2 64-layer config
    config = GPT2Config(n_layer=64)
    
    # Create model with 8 pipeline stages
    model = GPT2ForPipelineParallelism(config, num_stages=8)
    
    print(f"GPT-2 with {config.n_layer} layers")
    print(f"Hidden size: {config.n_embd}, Heads: {config.n_head}")
    print(f"Partitioned into {model.num_stages} stages:")
    
    for stage_id, stage in enumerate(model.stages):
        print(f"  Stage {stage_id}: Layers {stage.layer_range}, "
              f"{len(stage.h)} blocks")
    
    # Test stage 0
    stage_0 = model.get_stage(0)
    input_ids = torch.randint(0, config.vocab_size, (2, 128))  # Batch=2, seq=128
    output = stage_0(input_ids)
    
    print(f"\nStage 0 output shape: {output.shape}")
    print(f"Expected: [2, 128, {config.n_embd}]")
