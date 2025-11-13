"""
Stage partitioning for Chimera pipeline parallelism.
Distributes model layers across D pipeline stages with memory estimation.
"""

import math
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MemoryEstimate:
    """Memory estimates for a pipeline stage"""
    weight_memory_mb: float  # M_θ in MB
    activation_memory_mb: float  # M_a in MB (per micro-batch)
    peak_activation_mb: float  # Peak M_a during schedule


class StagePartitioner:
    """
    Partitions Transformer model layers across D pipeline stages.
    Default strategy: Even distribution with embedding on stage 0.
    """
    
    def __init__(self, num_stages: int, model_config: dict):
        """
        Args:
            num_stages: Number of pipeline stages (D)
            model_config: Dict with keys:
                - num_layers: Total transformer blocks
                - hidden_size: Hidden dimension
                - vocab_size: Vocabulary size
                - num_attention_heads: Attention heads
                - intermediate_size: MLP intermediate size
                - max_sequence_length: Max seq length
                - dtype_bytes: 2 for fp16, 4 for fp32
        """
        if num_stages % 2 != 0:
            raise ValueError("num_stages must be even for bidirectional pipelines")
        
        self.num_stages = num_stages
        self.config = model_config
        self.dtype_bytes = model_config.get('dtype_bytes', 2)  # fp16 default
        
    def partition_even_blocks(self) -> List[Tuple[int, int]]:
        """
        Even partition: Distribute layers evenly across stages.
        
        Returns:
            List of (start_layer, end_layer) tuples for each stage.
            Stage 0 includes embedding layer implicitly.
        
        Example: 48 layers, 4 stages → [(0,12), (12,24), (24,36), (36,48)]
        """
        total_layers = self.config['num_layers']
        
        if total_layers % self.num_stages != 0:
            # Handle uneven split
            return self._partition_uneven(total_layers)
        
        layers_per_stage = total_layers // self.num_stages
        partitions = []
        
        for stage_id in range(self.num_stages):
            start_layer = stage_id * layers_per_stage
            end_layer = start_layer + layers_per_stage
            partitions.append((start_layer, end_layer))
        
        return partitions
    
    def _partition_uneven(self, total_layers: int) -> List[Tuple[int, int]]:
        """
        Handle cases where layers don't divide evenly.
        Distribute remainder across first stages.
        """
        base_layers = total_layers // self.num_stages
        remainder = total_layers % self.num_stages
        
        partitions = []
        current_layer = 0
        
        for stage_id in range(self.num_stages):
            # Give extra layer to first 'remainder' stages
            num_layers = base_layers + (1 if stage_id < remainder else 0)
            partitions.append((current_layer, current_layer + num_layers))
            current_layer += num_layers
        
        return partitions
    
    def partition_custom(self, layer_assignment: List[int]) -> List[Tuple[int, int]]:
        """
        Custom partition based on provided layer counts.
        
        Args:
            layer_assignment: List of layer counts per stage
                              Must sum to total_layers
        
        Returns:
            List of (start_layer, end_layer) tuples
        """
        if len(layer_assignment) != self.num_stages:
            raise ValueError(f"Assignment must have {self.num_stages} entries")
        
        if sum(layer_assignment) != self.config['num_layers']:
            raise ValueError("Layer assignment must sum to total layers")
        
        partitions = []
        current_layer = 0
        
        for num_layers in layer_assignment:
            partitions.append((current_layer, current_layer + num_layers))
            current_layer += num_layers
        
        return partitions
    
    def estimate_memory(self, stage_id: int, layer_range: Tuple[int, int],
                       micro_batch_size: int) -> MemoryEstimate:
        """
        Estimate memory consumption for a pipeline stage.
        
        Args:
            stage_id: Stage identifier (0 to D-1)
            layer_range: (start_layer, end_layer) for this stage
            micro_batch_size: Micro-batch size B
        
        Returns:
            MemoryEstimate with weight and activation memory
        """
        # Weight memory (M_θ)
        weight_memory = self._compute_weight_memory(stage_id, layer_range)
        
        # Activation memory (M_a) per micro-batch
        activation_memory = self._compute_activation_memory(
            layer_range, micro_batch_size
        )
        
        # Peak activation memory during schedule
        # Chimera: [(D/2 + 1) * M_a, D * M_a] across workers
        peak_activation = self._estimate_peak_activation(
            activation_memory, stage_id
        )
        
        return MemoryEstimate(
            weight_memory_mb=weight_memory,
            activation_memory_mb=activation_memory,
            peak_activation_mb=peak_activation
        )
    
    def _compute_weight_memory(self, stage_id: int, 
                              layer_range: Tuple[int, int]) -> float:
        """
        Compute weight memory M_θ in MB.
        
        Stage 0: Embedding + transformer blocks
        Other stages: Only transformer blocks
        
        Transformer block parameters:
        - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
        - MLP: 2 * hidden_size * intermediate_size
        - LayerNorm: 2 * hidden_size (x2 for two norms)
        """
        hidden_size = self.config['hidden_size']
        intermediate_size = self.config['intermediate_size']
        num_blocks = layer_range[1] - layer_range[0]
        
        # Transformer block parameters
        attention_params = 4 * hidden_size * hidden_size
        mlp_params = 2 * hidden_size * intermediate_size
        layernorm_params = 2 * 2 * hidden_size  # Two LayerNorms
        
        params_per_block = attention_params + mlp_params + layernorm_params
        total_params = num_blocks * params_per_block
        
        # Embedding parameters (stage 0 only)
        if stage_id == 0:
            vocab_size = self.config['vocab_size']
            embedding_params = vocab_size * hidden_size  # Token embeddings
            position_params = self.config['max_sequence_length'] * hidden_size
            total_params += embedding_params + position_params
        
        # Convert to MB: params * dtype_bytes / (1024^2)
        # Note: Chimera stores 2 * M_θ per worker (two stage replicas)
        weight_memory_mb = (2 * total_params * self.dtype_bytes) / (1024 ** 2)
        
        return weight_memory_mb
    
    def _compute_activation_memory(self, layer_range: Tuple[int, int],
                                   micro_batch_size: int) -> float:
        """
        Compute activation memory M_a per micro-batch in MB.
        
        Activations stored per layer:
        - Attention outputs: [B, seq_len, hidden_size]
        - MLP outputs: [B, seq_len, hidden_size]
        - Intermediate states: [B, seq_len, intermediate_size]
        """
        B = micro_batch_size
        seq_len = self.config['max_sequence_length']
        hidden_size = self.config['hidden_size']
        intermediate_size = self.config['intermediate_size']
        num_blocks = layer_range[1] - layer_range[0]
        
        # Per-layer activation size
        attention_activations = B * seq_len * hidden_size
        mlp_activations = B * seq_len * hidden_size
        intermediate_activations = B * seq_len * intermediate_size
        
        activations_per_layer = (
            attention_activations + mlp_activations + intermediate_activations
        )
        
        total_activations = num_blocks * activations_per_layer
        
        # Convert to MB
        activation_memory_mb = (total_activations * self.dtype_bytes) / (1024 ** 2)
        
        return activation_memory_mb
    
    def _estimate_peak_activation(self, base_activation_mb: float,
                                  stage_id: int) -> float:
        """
        Estimate peak activation memory during Chimera schedule.
        
        From paper: Activation memory per worker in range
        [(D/2 + 1) * M_a, D * M_a]
        
        Conservative estimate: Use upper bound D * M_a
        """
        # Number of in-flight micro-batches at peak
        # Chimera: Up to D micro-batches can be in-flight
        max_in_flight = self.num_stages
        
        peak_memory = max_in_flight * base_activation_mb
        
        return peak_memory
    
    def get_memory_profile(self, partitions: List[Tuple[int, int]],
                          micro_batch_size: int) -> Dict[int, MemoryEstimate]:
        """
        Get memory estimates for all stages.
        
        Args:
            partitions: Layer ranges from partition_even_blocks()
            micro_batch_size: Micro-batch size B
        
        Returns:
            Dict mapping stage_id to MemoryEstimate
        """
        memory_profile = {}
        
        for stage_id, layer_range in enumerate(partitions):
            memory_profile[stage_id] = self.estimate_memory(
                stage_id, layer_range, micro_batch_size
            )
        
        return memory_profile
    
    def validate_memory_budget(self, partitions: List[Tuple[int, int]],
                              micro_batch_size: int,
                              memory_budget_gb: float) -> bool:
        """
        Check if partition fits within GPU memory budget.
        
        Args:
            partitions: Layer ranges
            micro_batch_size: Micro-batch size B
            memory_budget_gb: Available GPU memory in GB
        
        Returns:
            True if all stages fit in memory
        """
        memory_profile = self.get_memory_profile(partitions, micro_batch_size)
        
        for stage_id, mem_est in memory_profile.items():
            total_memory_mb = (
                mem_est.weight_memory_mb + mem_est.peak_activation_mb
            )
            total_memory_gb = total_memory_mb / 1024
            
            if total_memory_gb > memory_budget_gb:
                print(f"Stage {stage_id} exceeds memory: "
                      f"{total_memory_gb:.2f} GB > {memory_budget_gb:.2f} GB")
                return False
        
        return True


# Example usage
if __name__ == "__main__":
    # BERT-48 configuration
    bert_config = {
        'num_layers': 48,
        'hidden_size': 1024,
        'vocab_size': 30522,
        'num_attention_heads': 16,
        'intermediate_size': 4096,
        'max_sequence_length': 512,
        'dtype_bytes': 2  # fp16
    }
    
    partitioner = StagePartitioner(num_stages=4, model_config=bert_config)
    
    # Even partition
    partitions = partitioner.partition_even_blocks()
    print("Stage partitions:", partitions)
    
    # Memory estimates
    memory_profile = partitioner.get_memory_profile(partitions, micro_batch_size=4)
    
    for stage_id, mem_est in memory_profile.items():
        print(f"\nStage {stage_id}:")
        print(f"  Weight memory: {mem_est.weight_memory_mb:.2f} MB")
        print(f"  Activation memory (per micro-batch): {mem_est.activation_memory_mb:.2f} MB")
        print(f"  Peak activation: {mem_est.peak_activation_mb:.2f} MB")
    
    # Validate memory budget
    is_valid = partitioner.validate_memory_budget(
        partitions, micro_batch_size=4, memory_budget_gb=16.0
    )
    print(f"\nFits in 16GB memory: {is_valid}")
