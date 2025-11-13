"""Demonstrate BERT-48 pipeline partitioning"""
import torch
from chimera.models import BertConfig, BertForPipelineParallelism
from chimera.engine import StagePartitioner

print("Chimera Model Pipeline Demonstration")
print("="*60)

# Create BERT-48 model
config = BertConfig(
    num_hidden_layers=48,
    hidden_size=1024,
    vocab_size=30522,
    num_attention_heads=16,
    intermediate_size=4096,
    max_position_embeddings=512
)

print(f"\nModel Configuration:")
print(f"  Name: BERT-48")
print(f"  Total layers: {config.num_hidden_layers}")
print(f"  Hidden size: {config.hidden_size}")
print(f"  Parameters: ~669M")

# Partition into 4 stages
num_stages = 4
model = BertForPipelineParallelism(config, num_stages=num_stages)

print(f"\nPipeline Partitioning:")
print(f"  Number of stages: {num_stages}")
for i, stage in enumerate(model.stages):
    print(f"  Stage {i}: layers {stage.layer_range[0]}-{stage.layer_range[1]} "
          f"({stage.layer_range[1] - stage.layer_range[0]} blocks)")

# Estimate memory per stage
partitioner = StagePartitioner(
    num_stages=num_stages,
    model_config={
        'num_layers': 48,
        'hidden_size': 1024,
        'vocab_size': 30522,
        'num_attention_heads': 16,
        'intermediate_size': 4096,
        'max_sequence_length': 512,
        'dtype_bytes': 2
    }
)

partitions = partitioner.partition_even_blocks()
memory_profile = partitioner.get_memory_profile(partitions, micro_batch_size=4)

print(f"\nMemory Profile (B=4 micro-batch):")
for stage_id, mem in memory_profile.items():
    total_mb = mem.weight_memory_mb + mem.peak_activation_mb
    print(f"  Stage {stage_id}: {total_mb:.2f} MB "
          f"(weights: {mem.weight_memory_mb:.2f} MB, "
          f"activations: {mem.peak_activation_mb:.2f} MB)")

# Test data flow through first two stages
print(f"\nTesting Data Flow:")
stage_0 = model.get_stage(0)
stage_1 = model.get_stage(1)

input_ids = torch.randint(0, config.vocab_size, (2, 128))
print(f"  Input shape: {input_ids.shape}")

output_0 = stage_0(input_ids)
print(f"  Stage 0 output: {output_0.shape}")

output_1 = stage_1(output_0)
print(f"  Stage 1 output: {output_1.shape}")

print(f"\n✅ Pipeline data flow successful!")
