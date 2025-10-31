"""
BERT-48 model for Chimera pipeline parallelism.
669M parameter model with 48 Transformer layers.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class BertConfig:
    """Configuration for BERT-48 model"""
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 1024,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps


class BertEmbeddings(nn.Module):
    """BERT embedding layer (token + position)"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_length]
        
        Returns:
            embeddings: [batch_size, seq_length, hidden_size]
        """
        seq_length = input_ids.size(1)
        
        # Position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Token + position embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertSelfAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} not divisible by "
                f"num heads {config.num_attention_heads}"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape for multi-head attention.
        [batch, seq, hidden] -> [batch, heads, seq, head_size]
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_length, hidden_size]
        
        Returns:
            context: [batch, seq_length, hidden_size]
        """
        # Q, K, V projections
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Attention probabilities
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Context
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        return context_layer


class BertSelfOutput(nn.Module):
    """Output projection and residual for attention"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Complete attention block"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_output = self.self(hidden_states)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertMLP(nn.Module):
    """MLP/FFN block"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # FFN
        intermediate = self.dense_1(hidden_states)
        intermediate = nn.functional.gelu(intermediate)
        
        output = self.dense_2(intermediate)
        output = self.dropout(output)
        
        # Residual + LayerNorm
        output = self.LayerNorm(output + hidden_states)
        
        return output


class BertBlock(nn.Module):
    """Single BERT Transformer block"""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.mlp = BertMLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states)
        layer_output = self.mlp(attention_output)
        return layer_output


class BertStage(nn.Module):
    """
    Pipeline stage containing subset of BERT layers.
    Stage 0: Embeddings + first K blocks
    Other stages: K blocks each
    """
    
    def __init__(
        self,
        config: BertConfig,
        stage_id: int,
        layer_range: Tuple[int, int]
    ):
        """
        Args:
            config: BERT configuration
            stage_id: Stage identifier (0 to D-1)
            layer_range: (start_layer, end_layer) for this stage
        """
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.layer_range = layer_range
        
        # Stage 0: Include embeddings
        if stage_id == 0:
            self.embeddings = BertEmbeddings(config)
        else:
            self.embeddings = None
        
        # Transformer blocks
        start_layer, end_layer = layer_range
        self.blocks = nn.ModuleList([
            BertBlock(config)
            for _ in range(start_layer, end_layer)
        ])
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_data: [batch, seq_length] for stage 0 (input_ids)
                       [batch, seq_length, hidden_size] for other stages
        
        Returns:
            hidden_states: [batch, seq_length, hidden_size]
        """
        # Stage 0: Embedding layer
        if self.stage_id == 0:
            hidden_states = self.embeddings(input_data)
        else:
            hidden_states = input_data
        
        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states)
        
        return hidden_states


class BertForPipelineParallelism(nn.Module):
    """
    Complete BERT-48 model partitioned for pipeline parallelism.
    """
    
    def __init__(
        self,
        config: BertConfig,
        num_stages: int,
        partitions: Optional[list] = None
    ):
        """
        Args:
            config: BERT configuration
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
                    'num_layers': config.num_hidden_layers,
                    'hidden_size': config.hidden_size,
                    'vocab_size': config.vocab_size,
                    'num_attention_heads': config.num_attention_heads,
                    'intermediate_size': config.intermediate_size,
                    'max_sequence_length': config.max_position_embeddings,
                    'dtype_bytes': 2
                }
            )
            partitions = partitioner.partition_even_blocks()
        
        # Create stages
        self.stages = nn.ModuleList([
            BertStage(config, stage_id, layer_range)
            for stage_id, layer_range in enumerate(partitions)
        ])
    
    def get_stage(self, stage_id: int) -> BertStage:
        """Get specific pipeline stage"""
        return self.stages[stage_id]


# Example usage
if __name__ == "__main__":
    # Create BERT-48 config
    config = BertConfig(num_hidden_layers=48)
    
    # Create model with 4 pipeline stages
    model = BertForPipelineParallelism(config, num_stages=4)
    
    print(f"BERT-48 with {config.num_hidden_layers} layers")
    print(f"Partitioned into {model.num_stages} stages:")
    
    for stage_id, stage in enumerate(model.stages):
        print(f"  Stage {stage_id}: Layers {stage.layer_range}, "
              f"{len(stage.blocks)} blocks")
    
    # Test stage 0
    stage_0 = model.get_stage(0)
    input_ids = torch.randint(0, config.vocab_size, (2, 128))  # Batch=2, seq=128
    output = stage_0(input_ids)
    
    print(f"\nStage 0 output shape: {output.shape}")
