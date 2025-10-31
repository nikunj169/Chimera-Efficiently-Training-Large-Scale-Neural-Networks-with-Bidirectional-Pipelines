"""
Comprehensive test suite for Person A's implementation.
Validates all engine components and model wrappers.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chimera.engine.partition import StagePartitioner, MemoryEstimate
from chimera.engine.schedule import BidirectionalSchedule, ScheduleType, BubbleStats
from chimera.engine.runtime import StageWorker, MicroBatchContext
from chimera.engine.recompute import ActivationCheckpointing
from chimera.models.bert48 import BertConfig, BertStage, BertForPipelineParallelism
from chimera.models.gpt2_64 import GPT2Config, GPT2Stage, GPT2ForPipelineParallelism


class Colors:
    """ANSI color codes for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with color"""
    status = f"{Colors.GREEN}✓ PASSED{Colors.END}" if passed else f"{Colors.RED}✗ FAILED{Colors.END}"
    print(f"{status} - {name}")
    if details:
        print(f"  {details}")


def test_partition():
    """Test partition.py"""
    print(f"\n{Colors.BLUE}Testing partition.py{Colors.END}")
    
    # Test 1: Even partition
    try:
        config = {
            'num_layers': 48,
            'hidden_size': 1024,
            'vocab_size': 30522,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_sequence_length': 512,
            'dtype_bytes': 2
        }
        
        partitioner = StagePartitioner(num_stages=4, model_config=config)
        partitions = partitioner.partition_even_blocks()
        
        # Validate
        assert len(partitions) == 4, "Should have 4 partitions"
        assert partitions == [(0, 12), (12, 24), (24, 36), (36, 48)], "Incorrect partition"
        
        print_test("Even partition (48 layers → 4 stages)", True, 
                  f"Partitions: {partitions}")
    except Exception as e:
        print_test("Even partition", False, str(e))
    
    # Test 2: Memory estimation
    try:
        memory_profile = partitioner.get_memory_profile(partitions, micro_batch_size=4)
        
        assert len(memory_profile) == 4, "Should have 4 memory estimates"
        assert all(isinstance(v, MemoryEstimate) for v in memory_profile.values()), "Wrong type"
        
        # Stage 0 should have more memory (embeddings)
        assert memory_profile[0].weight_memory_mb > memory_profile[1].weight_memory_mb, \
            "Stage 0 should have more weight memory"
        
        print_test("Memory estimation", True,
                  f"Stage 0: {memory_profile[0].weight_memory_mb:.2f} MB weights, "
                  f"{memory_profile[0].peak_activation_mb:.2f} MB peak activations")
    except Exception as e:
        print_test("Memory estimation", False, str(e))
    
    # Test 3: Uneven partition
    try:
        partitioner2 = StagePartitioner(num_stages=4, model_config={**config, 'num_layers': 50})
        partitions2 = partitioner2.partition_even_blocks()
        
        # Should distribute remainder: 13, 13, 12, 12
        layer_counts = [end - start for start, end in partitions2]
        assert sum(layer_counts) == 50, "Should sum to 50 layers"
        assert max(layer_counts) - min(layer_counts) <= 1, "Should be balanced"
        
        print_test("Uneven partition (50 layers → 4 stages)", True,
                  f"Layer counts: {layer_counts}")
    except Exception as e:
        print_test("Uneven partition", False, str(e))
    
    # Test 4: Memory budget validation
    try:
        is_valid = partitioner.validate_memory_budget(partitions, micro_batch_size=4, 
                                                      memory_budget_gb=16.0)
        
        print_test("Memory budget validation", True,
                  f"Fits in 16GB: {is_valid}")
    except Exception as e:
        print_test("Memory budget validation", False, str(e))


def test_schedule():
    """Test schedule.py"""
    print(f"\n{Colors.BLUE}Testing schedule.py{Colors.END}")
    
    # Test 1: Base schedule (N=D)
    try:
        D, N = 4, 4
        scheduler = BidirectionalSchedule(D, N)
        schedule = scheduler.build_schedule(ScheduleType.BASE)
        
        # Validate
        assert len(schedule) == D, f"Should have {D} workers"
        assert all(len(schedule[i]) > 0 for i in range(D)), "All workers should have slots"
        
        # Check no conflicts
        for rank in range(D):
            times = [slot.time for slot in schedule[rank]]
            assert len(times) == len(set(times)), f"Time conflict at rank {rank}"
        
        # Count UNIQUE micro-batches (not total operations)
        unique_forward_mbs = set()
        unique_backward_mbs = set()
        for slots in schedule.values():
            for slot in slots:
                if slot.operation == 'forward':
                    unique_forward_mbs.add(slot.micro_batch_id)
                elif slot.operation == 'backward':
                    unique_backward_mbs.add(slot.micro_batch_id)
        
        assert len(unique_forward_mbs) == N, f"Should have {N} unique forward micro-batches, got {len(unique_forward_mbs)}"
        assert len(unique_backward_mbs) == N, f"Should have {N} unique backward micro-batches, got {len(unique_backward_mbs)}"
        
        print_test(f"Base schedule (D={D}, N={N})", True,
                  f"{len(unique_forward_mbs)} unique forward micro-batches, {len(unique_backward_mbs)} unique backward micro-batches")
    except Exception as e:
        print_test("Base schedule", False, str(e))
    
    # Test 2: Bubble statistics
    try:
        stats = scheduler.compute_bubble_stats()
        
        expected_bubbles = D - 2  # Should be 2 for D=4
        assert stats.total_bubbles == expected_bubbles, \
            f"Expected {expected_bubbles} bubbles, got {stats.total_bubbles}"
        
        expected_ratio = (D - 2) / (2 * N + D - 2)
        assert abs(stats.bubble_ratio - expected_ratio) < 1e-6, "Wrong bubble ratio"
        
        print_test("Bubble statistics", True,
                  f"Total bubbles: {stats.total_bubbles}, Ratio: {stats.bubble_ratio:.3f}")
    except Exception as e:
        print_test("Bubble statistics", False, str(e))
    
    # Test 3: N < D handling
    try:
        scheduler2 = BidirectionalSchedule(D=4, N=2)
        schedule2 = scheduler2.build_schedule()
        
        total_ops = sum(len(slots) for slots in schedule2.values())
        assert total_ops > 0, "Should have operations for N < D"
        
        # Check unique micro-batches
        unique_mbs = set()
        for slots in schedule2.values():
            for slot in slots:
                unique_mbs.add(slot.micro_batch_id)
        
        assert len(unique_mbs) == 2, f"Should process 2 unique micro-batches, got {len(unique_mbs)}"
        
        print_test("N < D handling (D=4, N=2)", True,
                  f"{len(unique_mbs)} unique micro-batches, {total_ops} total operations")
    except Exception as e:
        print_test("N < D handling", False, str(e))
    
    # Test 4: Direct concatenation (N > D)
    try:
        scheduler3 = BidirectionalSchedule(D=4, N=8)
        schedule3 = scheduler3.build_schedule(ScheduleType.DIRECT_CONCAT)
        
        # Count unique forward micro-batches
        unique_forward_mbs = set()
        for slots in schedule3.values():
            for slot in slots:
                if slot.operation == 'forward':
                    unique_forward_mbs.add(slot.micro_batch_id)
        
        assert len(unique_forward_mbs) == 8, f"Should have 8 unique forward micro-batches, got {len(unique_forward_mbs)}"
        
        print_test("Direct concatenation (D=4, N=8)", True,
                  f"{len(unique_forward_mbs)} unique forward micro-batches")
    except Exception as e:
        print_test("Direct concatenation", False, str(e))
    
    # Test 5: Forward doubling (N > D)
    try:
        scheduler4 = BidirectionalSchedule(D=4, N=8)
        schedule4 = scheduler4.build_schedule(ScheduleType.FORWARD_DOUBLING)
        
        # Check for recompute flags
        requires_recompute = sum(1 for slots in schedule4.values() 
                                for slot in slots if slot.requires_recompute)
        
        assert requires_recompute > 0, "Should have recompute slots"
        
        print_test("Forward doubling (D=4, N=8)", True,
                  f"{requires_recompute} slots require recompute")
    except Exception as e:
        print_test("Forward doubling", False, str(e))
    
    # Test 6: Eager sync stages
    try:
        eager_stages = scheduler.get_eager_sync_stages()
        
        assert eager_stages == {0, 3}, f"Expected {{0, 3}}, got {eager_stages}"
        
        print_test("Eager sync stages", True,
                  f"Edge stages: {eager_stages}")
    except Exception as e:
        print_test("Eager sync stages", False, str(e))


def test_recompute():
    """Test recompute.py"""
    print(f"\n{Colors.BLUE}Testing recompute.py{Colors.END}")
    
    # Test 1: Checkpoint function
    try:
        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 128)
            
            def forward(self, x):
                return self.linear(x)
        
        module = SimpleModule()
        x = torch.randn(2, 128, requires_grad=True)
        
        # Checkpoint forward
        output = ActivationCheckpointing.checkpoint_function(module, x)
        
        assert output.shape == x.shape, "Output shape mismatch"
        assert output.requires_grad, "Output should require grad"
        
        # Test backward
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "Input should have gradient"
        
        print_test("Checkpoint function", True,
                  f"Forward and backward successful")
    except Exception as e:
        print_test("Checkpoint function", False, str(e))
    
    # Test 2: Estimate overhead (FIXED)
    try:
        # Test with small number of blocks
        overhead = ActivationCheckpointing.estimate_recompute_overhead(num_checkpointed_blocks=1)
        
        assert overhead >= 1.0, "Overhead should be >= 1.0"
        assert overhead < 2.0, f"Overhead should be < 2.0, got {overhead}"
        
        # Test with zero blocks
        overhead_zero = ActivationCheckpointing.estimate_recompute_overhead(num_checkpointed_blocks=0)
        assert overhead_zero == 1.0, "Zero blocks should have no overhead"
        
        print_test("Recompute overhead estimation", True,
                  f"Overhead: {overhead:.2f}x (1 block), {overhead_zero:.2f}x (0 blocks)")
    except Exception as e:
        print_test("Recompute overhead estimation", False, str(e))
    
    # Test 3: Memory savings estimation
    try:
        savings = ActivationCheckpointing.estimate_memory_savings(
            activation_memory_mb=100.0,
            checkpoint_ratio=0.5
        )
        
        assert savings > 0, "Should have positive savings"
        assert savings < 100.0, "Savings should be less than total"
        
        print_test("Memory savings estimation", True,
                  f"Savings: {savings:.2f} MB from 100 MB")
    except Exception as e:
        print_test("Memory savings estimation", False, str(e))


def test_bert_model():
    """Test BERT model"""
    print(f"\n{Colors.BLUE}Testing BERT-48 model{Colors.END}")
    
    # Test 1: Model creation
    try:
        config = BertConfig(num_hidden_layers=48)
        model = BertForPipelineParallelism(config, num_stages=4)
        
        assert len(model.stages) == 4, "Should have 4 stages"
        
        print_test("BERT model creation", True,
                  f"{config.num_hidden_layers} layers → {len(model.stages)} stages")
    except Exception as e:
        print_test("BERT model creation", False, str(e))
    
    # Test 2: Stage 0 forward
    try:
        stage_0 = model.get_stage(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        
        output = stage_0(input_ids)
        
        expected_shape = (2, 128, config.hidden_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print_test("BERT stage 0 forward", True,
                  f"Output shape: {output.shape}")
    except Exception as e:
        print_test("BERT stage 0 forward", False, str(e))
    
    # Test 3: Stage 1 forward
    try:
        stage_1 = model.get_stage(1)
        hidden_states = torch.randn(2, 128, config.hidden_size)
        
        output = stage_1(hidden_states)
        
        assert output.shape == hidden_states.shape, "Output shape should match input"
        
        print_test("BERT stage 1 forward", True,
                  f"Output shape: {output.shape}")
    except Exception as e:
        print_test("BERT stage 1 forward", False, str(e))
    
    # Test 4: Backward pass
    try:
        stage_0 = model.get_stage(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        
        output = stage_0(input_ids)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None for p in stage_0.parameters())
        assert has_grads, "Should have gradients"
        
        print_test("BERT backward pass", True,
                  "Gradients computed successfully")
    except Exception as e:
        print_test("BERT backward pass", False, str(e))


def test_gpt2_model():
    """Test GPT-2 model"""
    print(f"\n{Colors.BLUE}Testing GPT-2 64-layer model{Colors.END}")
    
    # Test 1: Model creation
    try:
        config = GPT2Config(n_layer=64)
        model = GPT2ForPipelineParallelism(config, num_stages=8)
        
        assert len(model.stages) == 8, "Should have 8 stages"
        
        print_test("GPT-2 model creation", True,
                  f"{config.n_layer} layers → {len(model.stages)} stages")
    except Exception as e:
        print_test("GPT-2 model creation", False, str(e))
    
    # Test 2: Stage 0 forward
    try:
        stage_0 = model.get_stage(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        
        output = stage_0(input_ids)
        
        expected_shape = (2, 128, config.n_embd)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        print_test("GPT-2 stage 0 forward", True,
                  f"Output shape: {output.shape}")
    except Exception as e:
        print_test("GPT-2 stage 0 forward", False, str(e))
    
    # Test 3: Stage forward with causal mask
    try:
        stage_1 = model.get_stage(1)
        hidden_states = torch.randn(2, 128, config.n_embd)
        
        output = stage_1(hidden_states)
        
        assert output.shape == hidden_states.shape, "Output shape should match input"
        
        print_test("GPT-2 stage forward", True,
                  f"Output shape: {output.shape}")
    except Exception as e:
        print_test("GPT-2 stage forward", False, str(e))
    
    # Test 4: Backward pass
    try:
        stage_0 = model.get_stage(0)
        input_ids = torch.randint(0, config.vocab_size, (2, 128))
        
        output = stage_0(input_ids)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        has_grads = any(p.grad is not None for p in stage_0.parameters())
        assert has_grads, "Should have gradients"
        
        print_test("GPT-2 backward pass", True,
                  "Gradients computed successfully")
    except Exception as e:
        print_test("GPT-2 backward pass", False, str(e))


def test_integration():
    """Integration test: Full pipeline with schedule"""
    print(f"\n{Colors.BLUE}Integration Tests{Colors.END}")
    
    # Test: Schedule + Model integration
    try:
        # Create small model
        config = BertConfig(num_hidden_layers=4, hidden_size=128, 
                           num_attention_heads=4, intermediate_size=512)
        model = BertForPipelineParallelism(config, num_stages=2)
        
        # Create schedule
        scheduler = BidirectionalSchedule(D=2, N=2)
        schedule = scheduler.build_schedule()
        
        # Verify schedule works with model stages
        assert len(model.stages) == 2, "Should have 2 stages"
        assert len(schedule) == 2, "Should have 2 workers"
        
        # Test data flow
        stage_0 = model.get_stage(0)
        stage_1 = model.get_stage(1)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        
        # Stage 0 → Stage 1
        hidden_0 = stage_0(input_ids)
        output = stage_1(hidden_0)
        
        assert output.shape == (1, 32, config.hidden_size), "Output shape mismatch"
        
        print_test("Schedule + Model integration", True,
                  f"2-stage pipeline successful")
    except Exception as e:
        print_test("Schedule + Model integration", False, str(e))


def main():
    """Run all tests"""
    print(f"\n{Colors.YELLOW}{'='*60}{Colors.END}")
    print(f"{Colors.YELLOW}Chimera Person A Implementation Test Suite{Colors.END}")
    print(f"{Colors.YELLOW}{'='*60}{Colors.END}")
    
    test_partition()
    test_schedule()
    test_recompute()
    test_bert_model()
    test_gpt2_model()
    test_integration()
    
    print(f"\n{Colors.YELLOW}{'='*60}{Colors.END}")
    print(f"{Colors.GREEN}Test suite completed!{Colors.END}")
    print(f"{Colors.YELLOW}{'='*60}{Colors.END}\n")


if __name__ == "__main__":
    main()
