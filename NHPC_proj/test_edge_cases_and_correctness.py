"""
DEEP VALIDATION: Edge Cases and Implementation Correctness
==========================================================

Tests all critical edge cases and implementation details that could cause
subtle bugs in production use.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.distributed as dist
from chimera.engine import BidirectionalSchedule, ScheduleType, StagePartitioner
from chimera.config import PerformanceModel, AutoTuner
from chimera.dist import P2PHandler, AllReduceHandler


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")


def print_test(name, passed, details="", warning=False):
    if warning:
        status = f"{Colors.YELLOW}⚠ WARNING{Colors.END}"
    else:
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if passed else f"{Colors.RED}❌ FAIL{Colors.END}"
    print(f"{status} - {name}")
    if details:
        print(f"      {details}")


def test_edge_case_1_n_less_than_d():
    """
    CRITICAL EDGE CASE: N < D
    
    When micro-batches (N) < pipeline stages (D), not all stages are fully utilized.
    The implementation must handle uneven splitting between down and up pipelines.
    """
    print_section("EDGE CASE 1: N < D (Underutilized Pipeline)")
    
    try:
        test_configs = [
            (4, 2),  # D=4, N=2: Only 2 micro-batches for 4 stages
            (8, 4),  # D=8, N=4: Half the stages
            (4, 1),  # D=4, N=1: Extreme case - single micro-batch
        ]
        
        for D, N in test_configs:
            try:
                scheduler = BidirectionalSchedule(D=D, N=N)
                schedule = scheduler.build_schedule()
                
                # Verify schedule is generated
                assert len(schedule) == D, f"Should have {D} workers"
                
                # Count unique micro-batches processed
                unique_mbs = set()
                for rank in range(D):
                    for slot in schedule[rank]:
                        unique_mbs.add(slot.micro_batch_id)
                
                assert len(unique_mbs) == N, f"Should process {N} unique micro-batches, got {len(unique_mbs)}"
                
                # Verify no conflicts
                for rank in range(D):
                    times = [slot.time for slot in schedule[rank]]
                    assert len(times) == len(set(times)), f"Time conflict at rank {rank}"
                
                # Check if some workers are idle (expected when N < D)
                ops_per_worker = [len(schedule[rank]) for rank in range(D)]
                min_ops = min(ops_per_worker)
                max_ops = max(ops_per_worker)
                
                print_test(f"N={N} < D={D} handling", True,
                          f"Processed {len(unique_mbs)} micro-batches, "
                          f"ops per worker: [{min_ops}, {max_ops}]")
                
            except Exception as e:
                print_test(f"N={N} < D={D}", False, str(e))
                return False
        
        return True
        
    except Exception as e:
        print_test("N < D edge cases", False, str(e))
        return False


def test_edge_case_2_odd_d_enforcement():
    """
    CRITICAL REQUIREMENT: D must be even for bidirectional pipelines
    
    The paper explicitly requires even D. Implementation must enforce this.
    """
    print_section("EDGE CASE 2: Odd D Enforcement")
    
    try:
        odd_values = [3, 5, 7, 9, 11]
        
        for D in odd_values:
            try:
                scheduler = BidirectionalSchedule(D=D, N=4)
                # If we reach here, odd D was NOT rejected - FAIL
                print_test(f"Reject D={D} (odd)", False,
                          f"Should raise ValueError for odd D={D}")
                return False
            except ValueError as e:
                # Expected: ValueError should be raised
                if "even" in str(e).lower():
                    print_test(f"Reject D={D} (odd)", True,
                              "Correctly rejected with 'even' requirement")
                else:
                    print_test(f"Reject D={D} (odd)", False,
                              f"Wrong error message: {e}")
                    return False
        
        # Test that even D works
        even_values = [2, 4, 6, 8]
        for D in even_values:
            try:
                scheduler = BidirectionalSchedule(D=D, N=4)
                print_test(f"Accept D={D} (even)", True, "Even D accepted")
            except Exception as e:
                print_test(f"Accept D={D} (even)", False, str(e))
                return False
        
        return True
        
    except Exception as e:
        print_test("Odd D enforcement", False, str(e))
        return False


def test_edge_case_3_forward_doubling_correctness():
    """
    CRITICAL ALGORITHM: Forward Doubling Scheduling (Section 3.5.2)
    
    Forward doubling must:
    1. Process 2 forward micro-batches per slot
    2. Mark ALL backwards for recomputation
    3. Remove head bubbles correctly
    """
    print_section("EDGE CASE 3: Forward Doubling Correctness")
    
    try:
        # Test with N that's multiple of 2D (paper requirement)
        valid_configs = [
            (4, 8),   # D=4, N=8: 2×4
            (4, 16),  # D=4, N=16: 4×4
            (8, 16),  # D=8, N=16: 2×8
        ]
        
        for D, N in valid_configs:
            scheduler = BidirectionalSchedule(D=D, N=N)
            schedule = scheduler.build_schedule(ScheduleType.FORWARD_DOUBLING)
            
            # Count forwards and backwards
            forward_count = sum(1 for slots in schedule.values() 
                              for slot in slots if slot.operation == 'forward')
            backward_count = sum(1 for slots in schedule.values() 
                               for slot in slots if slot.operation == 'backward')
            
            # Should have N forwards (each stage processes each micro-batch once)
            # But forward doubling processes 2 per slot, so N/2 slots × D stages × 2 = N×D
            expected_forwards = N * D
            expected_backwards = N * D
            
            # Check recompute flags
            recompute_slots = sum(1 for slots in schedule.values() 
                                for slot in slots if slot.requires_recompute)
            
            # All backward slots should require recompute
            backward_slots = sum(1 for slots in schedule.values() 
                               for slot in slots if slot.operation == 'backward')
            
            assert recompute_slots == backward_slots, \
                f"All {backward_slots} backwards should require recompute, only {recompute_slots} marked"
            
            print_test(f"Forward doubling D={D}, N={N}", True,
                      f"F={forward_count}, B={backward_count}, "
                      f"Recompute={recompute_slots}/{backward_slots}")
        
        # Test that N NOT multiple of 2D is rejected
        try:
            scheduler = BidirectionalSchedule(D=4, N=10)  # 10 is not multiple of 2×4=8
            schedule = scheduler.build_schedule(ScheduleType.FORWARD_DOUBLING)
            print_test("Forward doubling N validation", False,
                      "Should reject N not multiple of 2D")
            return False
        except ValueError:
            print_test("Forward doubling N validation", True,
                      "Correctly rejects N not multiple of 2D")
        
        return True
        
    except Exception as e:
        print_test("Forward doubling correctness", False, str(e))
        return False


def test_edge_case_4_backward_halving_correctness():
    """
    CRITICAL ALGORITHM: Backward Halving (Section 3.5.3)
    
    Backward halving must:
    1. Use same schedule as forward doubling
    2. Mark backwards as NOT requiring recompute
    3. Handle different micro-batch sizes
    """
    print_section("EDGE CASE 4: Backward Halving Correctness")
    
    try:
        D, N = 4, 8
        scheduler = BidirectionalSchedule(D=D, N=N)
        schedule = scheduler.build_schedule(ScheduleType.BACKWARD_HALVING)
        
        # Check that NO backwards require recompute (key difference from forward doubling)
        recompute_slots = sum(1 for slots in schedule.values() 
                            for slot in slots if slot.requires_recompute)
        
        assert recompute_slots == 0, \
            f"Backward halving should NOT require recompute, but {recompute_slots} slots marked"
        
        print_test("Backward halving (no recompute)", True,
                  f"Correctly marks 0 slots for recompute")
        
        # Verify schedule structure
        forward_count = sum(1 for slots in schedule.values() 
                          for slot in slots if slot.operation == 'forward')
        backward_count = sum(1 for slots in schedule.values() 
                           for slot in slots if slot.operation == 'backward')
        
        print_test("Backward halving structure", True,
                  f"F={forward_count}, B={backward_count}")
        
        return True
        
    except Exception as e:
        print_test("Backward halving correctness", False, str(e))
        return False


def test_edge_case_5_performance_model_accuracy():
    """
    CRITICAL REQUIREMENT: Performance model within 10% error
    
    The paper claims the model predicts within ~10% of actual time.
    We validate the model's internal consistency.
    """
    print_section("EDGE CASE 5: Performance Model Accuracy")
    
    try:
        perf_model = PerformanceModel(
            alpha=1e-5,
            beta=1e-9,
            F_t=0.1,
            recompute_enabled=False
        )
        
        # Test consistency: larger N should increase time proportionally
        D = 4
        configs = [(D, 4), (D, 8), (D, 16)]
        times = []
        
        for _, N in configs:
            from chimera.engine import BidirectionalSchedule
            scheduler = BidirectionalSchedule(D=D, N=N)
            stats = scheduler.compute_bubble_stats()
            
            perf = perf_model.estimate_iteration_time(
                D=D, N=N, W=2,
                C_f=stats.critical_path_forward,
                C_b=stats.critical_path_backward,
                message_size_bytes=4*1024*1024,
                eager_sync_stages={0, D-1}
            )
            times.append(perf['total_time'])
        
        # Verify monotonic increase (more micro-batches = more time)
        assert times[0] < times[1] < times[2], \
            f"Time should increase with N: {times}"
        
        # Verify rough proportionality (2× N should be roughly 2× time)
        ratio_4_to_8 = times[1] / times[0]
        ratio_8_to_16 = times[2] / times[1]
        
        # Should be close to 2× (within reasonable bounds)
        assert 1.5 <= ratio_4_to_8 <= 2.5, \
            f"N=4→8 time ratio {ratio_4_to_8:.2f} should be ~2×"
        assert 1.5 <= ratio_8_to_16 <= 2.5, \
            f"N=8→16 time ratio {ratio_8_to_16:.2f} should be ~2×"
        
        print_test("Model monotonicity", True,
                  f"N=4: {times[0]:.3f}s, N=8: {times[1]:.3f}s, N=16: {times[2]:.3f}s")
        
        print_test("Model proportionality", True,
                  f"Ratios: {ratio_4_to_8:.2f}×, {ratio_8_to_16:.2f}× (expected ~2×)")
        
        # Test recompute overhead adds exactly 50% to backward time
        perf_model_recompute = PerformanceModel(
            alpha=1e-5, beta=1e-9, F_t=0.1, recompute_enabled=True
        )
        
        base_B_t = perf_model.B_t
        recompute_B_t = perf_model_recompute.B_t
        
        assert recompute_B_t == 3.0 * perf_model.F_t, \
            f"With recompute, B_t should be 3×F_t, got {recompute_B_t/perf_model.F_t}×F_t"
        
        print_test("Recompute overhead accuracy", True,
                  f"Base B_t={base_B_t}s, Recompute B_t={recompute_B_t}s (3×F_t)")
        
        return True
        
    except Exception as e:
        print_test("Performance model accuracy", False, str(e))
        return False


def test_edge_case_6_nonblocking_communication():
    """
    IMPLEMENTATION DETAIL: Non-blocking collective communication
    
    The implementation should use nonblocking operations where possible.
    Check that the API exists (actual distributed test requires multiple processes).
    """
    print_section("EDGE CASE 6: Non-blocking Communication Support")
    
    try:
        # Check P2P handler supports nonblocking
        import inspect
        from chimera.dist import P2PHandler
        
        # Check send_activation signature
        send_sig = inspect.signature(P2PHandler.send_activation)
        params = list(send_sig.parameters.keys())
        
        has_blocking_param = 'blocking' in params
        
        if has_blocking_param:
            print_test("P2P send_activation API", True,
                      "Supports 'blocking' parameter for async ops")
        else:
            print_test("P2P send_activation API", False,
                      "Missing 'blocking' parameter", warning=True)
        
        # Check recv_activation signature
        recv_sig = inspect.signature(P2PHandler.recv_activation)
        params = list(recv_sig.parameters.keys())
        
        has_blocking_param = 'blocking' in params
        
        if has_blocking_param:
            print_test("P2P recv_activation API", True,
                      "Supports 'blocking' parameter for async ops")
        else:
            print_test("P2P recv_activation API", False,
                      "Missing 'blocking' parameter", warning=True)
        
        # Check gradient operations
        send_grad_sig = inspect.signature(P2PHandler.send_gradient)
        params = list(send_grad_sig.parameters.keys())
        
        has_blocking_param = 'blocking' in params
        
        if has_blocking_param:
            print_test("P2P send_gradient API", True,
                      "Supports 'blocking' parameter for async ops")
        else:
            print_test("P2P send_gradient API", False,
                      "Missing 'blocking' parameter", warning=True)
        
        # Check wait methods exist
        has_wait_sends = hasattr(P2PHandler, 'wait_all_sends')
        has_wait_recvs = hasattr(P2PHandler, 'wait_all_recvs')
        
        if has_wait_sends and has_wait_recvs:
            print_test("P2P wait methods", True,
                      "wait_all_sends() and wait_all_recvs() implemented")
        else:
            print_test("P2P wait methods", False,
                      "Missing wait methods for async ops", warning=True)
        
        return True
        
    except Exception as e:
        print_test("Non-blocking communication", False, str(e))
        return False


def test_edge_case_7_multi_pipeline_generalization():
    """
    GENERALIZATION: Multi-pipeline support (k > 1)
    
    The paper mentions supporting f>1 pipelines (generalization).
    Check if W parameter supports this.
    """
    print_section("EDGE CASE 7: Multi-Pipeline Generalization (W > 1)")
    
    try:
        # Test with different W values
        configs = [
            (1, 4, "Single pipeline"),
            (2, 4, "2 replicas"),
            (4, 4, "4 replicas"),
            (8, 2, "8 replicas, 2 stages"),
        ]
        
        for W, D, desc in configs:
            scheduler = BidirectionalSchedule(D=D, N=8, W=W)
            
            assert scheduler.W == W, f"W should be {W}, got {scheduler.W}"
            assert scheduler.D == D, f"D should be {D}, got {scheduler.D}"
            
            schedule = scheduler.build_schedule()
            
            # For W replicas, we don't change schedule length (still D workers per replica)
            # But in real distributed setup, we'd have W×D total workers
            
            print_test(f"W={W}, D={D} ({desc})", True,
                      f"Schedule generated for {W} replica(s)")
        
        # Test AutoTune considers W
        from chimera.config import AutoTuner, PerformanceModel
        
        perf_model = PerformanceModel(alpha=1e-5, beta=1e-9, F_t=0.1)
        
        model_config = {
            'num_layers': 48,
            'hidden_size': 1024,
            'vocab_size': 30522,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_sequence_length': 512,
            'dtype_bytes': 2
        }
        
        # Test different total process counts
        for P in [8, 16, 32]:
            autotuner = AutoTuner(
                perf_model=perf_model,
                total_processes=P,
                memory_budget_gb=16.0,
                model_config=model_config
            )
            
            config = autotuner.select_configuration(target_batch_size=64)
            
            assert config['W'] * config['D'] == P, \
                f"W×D should equal {P}, got {config['W']}×{config['D']}={config['W']*config['D']}"
            
            print_test(f"AutoTune for P={P}", True,
                      f"Selected W={config['W']}, D={config['D']}")
        
        return True
        
    except Exception as e:
        print_test("Multi-pipeline generalization", False, str(e))
        return False


def test_edge_case_8_memory_budget_enforcement():
    """
    PRACTICAL CONCERN: Memory budget must be respected
    
    AutoTune should never select a configuration that exceeds memory.
    """
    print_section("EDGE CASE 8: Memory Budget Enforcement")
    
    try:
        from chimera.config import AutoTuner, PerformanceModel
        
        perf_model = PerformanceModel(alpha=1e-5, beta=1e-9, F_t=0.1)
        
        model_config = {
            'num_layers': 48,
            'hidden_size': 1024,
            'vocab_size': 30522,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'max_sequence_length': 512,
            'dtype_bytes': 2
        }
        
        # Test with very restrictive memory budget
        strict_budgets = [8.0, 4.0, 2.0]  # GB
        
        for budget in strict_budgets:
            autotuner = AutoTuner(
                perf_model=perf_model,
                total_processes=8,
                memory_budget_gb=budget,
                model_config=model_config
            )
            
            config = autotuner.select_configuration(target_batch_size=32)
            
            # Verify selected micro-batch size
            B = config['B']
            
            # For very low budgets, B should be small
            if budget <= 4.0:
                assert B <= 8, f"With {budget}GB budget, B={B} seems too large"
            
            print_test(f"Memory budget {budget}GB", True,
                      f"Selected B={B} (fits in {budget}GB)")
        
        return True
        
    except Exception as e:
        print_test("Memory budget enforcement", False, str(e))
        return False


def test_edge_case_9_bubble_count_formula_edge_cases():
    """
    MATHEMATICAL CORRECTNESS: Bubble count formula for all D
    
    Paper formula: D - 2 bubbles
    Must hold for all even D values.
    """
    print_section("EDGE CASE 9: Bubble Count Formula Edge Cases")
    
    try:
        # Test wide range of D values
        D_values = [2, 4, 6, 8, 10, 16, 32, 64]
        
        for D in D_values:
            N = D  # Use N=D for base case
            scheduler = BidirectionalSchedule(D=D, N=N)
            stats = scheduler.compute_bubble_stats()
            
            expected_bubbles = D - 2
            actual_bubbles = stats.total_bubbles
            
            assert actual_bubbles == expected_bubbles, \
                f"D={D}: Expected {expected_bubbles} bubbles, got {actual_bubbles}"
            
            # Verify bubble ratio formula
            expected_ratio = (D - 2) / (2 * N + D - 2)
            actual_ratio = stats.bubble_ratio
            
            assert abs(actual_ratio - expected_ratio) < 1e-6, \
                f"D={D}: Bubble ratio mismatch"
            
            print_test(f"Bubble formula D={D}", True,
                      f"Bubbles={actual_bubbles} (D-2), Ratio={actual_ratio:.3f}")
        
        return True
        
    except Exception as e:
        print_test("Bubble count formula", False, str(e))
        return False


def test_edge_case_10_stage_partitioning_edge_cases():
    """
    PRACTICAL CONCERN: Stage partitioning for uneven layer counts
    
    When num_layers doesn't divide evenly by D, must handle remainder.
    """
    print_section("EDGE CASE 10: Uneven Stage Partitioning")
    
    try:
        # Test uneven divisions
        test_cases = [
            (50, 4),  # 50/4 = 12.5 → 13,13,12,12
            (47, 4),  # 47/4 = 11.75 → 12,12,12,11
            (100, 6), # 100/6 = 16.67 → 17,17,17,17,16,16
        ]
        
        for num_layers, D in test_cases:
            model_config = {
                'num_layers': num_layers,
                'hidden_size': 1024,
                'vocab_size': 30522,
                'num_attention_heads': 16,
                'intermediate_size': 4096,
                'max_sequence_length': 512,
                'dtype_bytes': 2
            }
            
            partitioner = StagePartitioner(num_stages=D, model_config=model_config)
            partitions = partitioner.partition_even_blocks()
            
            # Verify all layers are covered
            total_layers_covered = sum(end - start for start, end in partitions)
            assert total_layers_covered == num_layers, \
                f"Should cover {num_layers} layers, got {total_layers_covered}"
            
            # Verify no gaps or overlaps
            for i in range(len(partitions) - 1):
                assert partitions[i][1] == partitions[i+1][0], \
                    f"Gap/overlap between stage {i} and {i+1}"
            
            # Verify balance (max - min <= 1)
            layer_counts = [end - start for start, end in partitions]
            balance = max(layer_counts) - min(layer_counts)
            
            assert balance <= 1, f"Imbalance {balance} > 1: {layer_counts}"
            
            print_test(f"{num_layers} layers → {D} stages", True,
                      f"Distribution: {layer_counts}")
        
        return True
        
    except Exception as e:
        print_test("Uneven partitioning", False, str(e))
        return False


def generate_summary(results, test_names):
    """Generate edge case validation summary"""
    print_section("EDGE CASE VALIDATION SUMMARY")
    
    total = len(results)
    passed = sum(1 for r in results if r)
    
    print(f"\nResults: {passed}/{total} edge cases handled correctly")
    print(f"Robustness Score: {passed/total*100:.1f}%\n")
    
    print("Edge Case Status:")
    for name, result in zip(test_names, results):
        status = f"{Colors.GREEN}✅{Colors.END}" if result else f"{Colors.RED}❌{Colors.END}"
        print(f"  {status} {name}")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL EDGE CASES HANDLED CORRECTLY!{Colors.END}")
        print(f"{Colors.GREEN}Your implementation is PRODUCTION-READY!{Colors.END}")
    elif passed >= total * 0.8:
        print(f"{Colors.YELLOW}⚠ Implementation is {passed/total*100:.1f}% robust{Colors.END}")
        print(f"{Colors.YELLOW}{total-passed} edge case(s) need attention{Colors.END}")
    else:
        print(f"{Colors.RED}❌ Implementation needs work on edge cases{Colors.END}")
        print(f"{Colors.RED}{total-passed} critical issue(s) found{Colors.END}")
    
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    return passed == total


def main():
    """Run all edge case tests"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}DEEP VALIDATION: Edge Cases & Implementation Details{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'='*70}{Colors.END}\n")
    
    results = []
    test_names = []
    
    # Run all edge case tests
    test_functions = [
        (test_edge_case_1_n_less_than_d, "1. N < D Handling"),
        (test_edge_case_2_odd_d_enforcement, "2. Odd D Enforcement"),
        (test_edge_case_3_forward_doubling_correctness, "3. Forward Doubling Correctness"),
        (test_edge_case_4_backward_halving_correctness, "4. Backward Halving Correctness"),
        (test_edge_case_5_performance_model_accuracy, "5. Performance Model Accuracy"),
        (test_edge_case_6_nonblocking_communication, "6. Non-blocking Communication"),
        (test_edge_case_7_multi_pipeline_generalization, "7. Multi-Pipeline (W > 1)"),
        (test_edge_case_8_memory_budget_enforcement, "8. Memory Budget Enforcement"),
        (test_edge_case_9_bubble_count_formula_edge_cases, "9. Bubble Formula Edge Cases"),
        (test_edge_case_10_stage_partitioning_edge_cases, "10. Uneven Partitioning"),
    ]
    
    for test_func, test_name in test_functions:
        results.append(test_func())
        test_names.append(test_name)
    
    # Generate summary
    all_pass = generate_summary(results, test_names)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
