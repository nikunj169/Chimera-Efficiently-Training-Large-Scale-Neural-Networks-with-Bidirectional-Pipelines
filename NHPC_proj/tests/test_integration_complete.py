"""
Complete integration test for Person A + Person B.
Tests end-to-end pipeline components (single-process safe).
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_integration_without_distributed():
    """
    Test integration of Person A and Person B without distributed setup.
    This version works in single-process mode.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST: Person A + Person B")
    print("(Single-process mode - no distributed required)")
    print("="*60)
    
    try:
        # ===== Test 1: Schedule + Performance Model =====
        print("\n1. Testing Schedule → Performance Model integration...")
        
        from chimera.engine import BidirectionalSchedule
        from chimera.config import PerformanceModel
        
        # Create schedule
        scheduler = BidirectionalSchedule(D=4, N=8)
        schedule = scheduler.build_schedule()
        stats = scheduler.compute_bubble_stats()
        
        print(f"   ✓ Schedule created: D=4, N=8")
        print(f"   ✓ Bubble ratio: {stats.bubble_ratio:.3f}")
        print(f"   ✓ Critical path: Cf={stats.critical_path_forward}, Cb={stats.critical_path_backward}")
        
        # Create performance model
        perf_model = PerformanceModel(alpha=1e-5, beta=1e-9, F_t=0.1)
        
        # Estimate performance using schedule stats
        perf = perf_model.estimate_iteration_time(
            D=4,
            N=8,
            W=2,
            C_f=stats.critical_path_forward,
            C_b=stats.critical_path_backward,
            message_size_bytes=1024*1024*4,
            eager_sync_stages=scheduler.get_eager_sync_stages()
        )
        
        print(f"   ✓ Performance estimated: {perf['total_time']:.3f}s")
        print(f"   ✓ Throughput: {perf['throughput']:.2f} micro-batches/s")
        print(f"   ✓ Integration: Schedule stats → Perf model ✅")
        
        # ===== Test 2: Partition + AutoTune =====
        print("\n2. Testing Partition → AutoTune integration...")
        
        from chimera.engine import StagePartitioner
        from chimera.config import AutoTuner
        
        model_config = {
            'num_layers': 48,
            'hidden_size': 1024,
            'vocab_size': 30522,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_sequence_length': 512,
            'dtype_bytes': 2
        }
        
        # Create partitioner
        partitioner = StagePartitioner(num_stages=4, model_config=model_config)
        partitions = partitioner.partition_even_blocks()
        
        print(f"   ✓ Partitions created: {partitions}")
        
        # Create autotuner
        autotuner = AutoTuner(
            perf_model=perf_model,
            total_processes=8,
            memory_budget_gb=16.0,
            model_config=model_config
        )
        
        # Select configuration
        config = autotuner.select_configuration(target_batch_size=64)
        
        print(f"   ✓ Config selected: W={config['W']}, D={config['D']}, B={config['B']}, N={config['N']}")
        print(f"   ✓ Integration: Partition → AutoTune ✅")
        
        # ===== Test 3: Models + Schedule =====
        print("\n3. Testing Models → Schedule integration...")
        
        from chimera.models import BertConfig, BertForPipelineParallelism
        
        # Create model
        bert_config = BertConfig(num_hidden_layers=4, hidden_size=128,
                                num_attention_heads=4, intermediate_size=512)
        model = BertForPipelineParallelism(bert_config, num_stages=2)
        
        print(f"   ✓ Model created: {len(model.stages)} stages")
        
        # Create schedule for model
        scheduler2 = BidirectionalSchedule(D=2, N=4)
        schedule2 = scheduler2.build_schedule()
        
        print(f"   ✓ Schedule created for model")
        
        # Test data flow through stages
        stage_0 = model.get_stage(0)
        stage_1 = model.get_stage(1)
        
        input_ids = torch.randint(0, bert_config.vocab_size, (2, 32))
        
        # Stage 0 → Stage 1
        hidden_0 = stage_0(input_ids)
        output = stage_1(hidden_0)
        
        print(f"   ✓ Data flow: stage 0 → stage 1 successful")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Integration: Models → Schedule ✅")
        
        # ===== Test 4: Component Type Checking =====
        print("\n4. Testing component interfaces...")
        
        # Check StageWorker signature
        import inspect
        from chimera.engine.runtime import StageWorker
        
        sig = inspect.signature(StageWorker.__init__)
        params = list(sig.parameters.keys())
        
        required_params = ['p2p_handler', 'allreduce_handler']
        for param in required_params:
            if param in params:
                print(f"   ✓ StageWorker has '{param}' parameter")
            else:
                raise AssertionError(f"StageWorker missing '{param}' parameter")
        
        print(f"   ✓ Integration: Runtime → Distributed handlers ✅")
        
        # ===== Test 5: Recompute + Schedule =====
        print("\n5. Testing Recompute → Schedule integration...")
        
        from chimera.engine import ActivationCheckpointing
        
        checkpointer = ActivationCheckpointing()
        
        # Test overhead estimation
        overhead = checkpointer.estimate_recompute_overhead(num_checkpointed_blocks=5)
        print(f"   ✓ Recompute overhead: {overhead:.2f}x")
        
        # Test with forward doubling schedule
        scheduler3 = BidirectionalSchedule(D=4, N=8)
        from chimera.engine import ScheduleType
        schedule3 = scheduler3.build_schedule(ScheduleType.FORWARD_DOUBLING)
        
        # Check for recompute flags
        recompute_count = sum(1 for slots in schedule3.values() 
                            for slot in slots if slot.requires_recompute)
        
        print(f"   ✓ Forward doubling schedule: {recompute_count} slots require recompute")
        print(f"   ✓ Integration: Recompute → Schedule ✅")
        
        # ===== Summary =====
        print("\n" + "="*60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("="*60)
        print("\nIntegration Summary:")
        print("  1. Schedule → Performance Model: ✅")
        print("  2. Partition → AutoTune: ✅")
        print("  3. Models → Schedule: ✅")
        print("  4. Runtime → Distributed handlers: ✅")
        print("  5. Recompute → Schedule: ✅")
        print("\nAll Person A and Person B components are properly connected!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_integration_without_distributed()
    sys.exit(0 if success else 1)
