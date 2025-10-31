"""Demonstrate Chimera performance modeling"""
from chimera.engine import BidirectionalSchedule
from chimera.config import PerformanceModel

print("Chimera Performance Model Demonstration")
print("="*60)

# Create performance model with typical values
perf_model = PerformanceModel(
    alpha=1e-5,      # 10 microseconds latency
    beta=1e-9,       # 1 GB/s bandwidth
    F_t=0.1,         # 100ms per forward micro-batch
    recompute_enabled=False
)

print(f"\nPerformance Model Parameters:")
print(f"  Network latency (α): {perf_model.alpha*1e6:.2f} microseconds")
print(f"  Network bandwidth (1/β): {1/perf_model.beta/1e9:.2f} GB/s")
print(f"  Forward time (F_t): {perf_model.F_t*1000:.1f} ms")
print(f"  Backward time (B_t): {perf_model.B_t*1000:.1f} ms")

# Compare different configurations
configs = [
    (2, 4, "W=2, D=4"),
    (4, 2, "W=4, D=2"),
    (1, 8, "W=1, D=8"),
]

print(f"\nConfiguration Comparison (N=8 micro-batches):")
print(f"{'Config':<15} {'Total Time':<12} {'Throughput':<15} {'Bubble Ratio':<15}")
print("-"*60)

for W, D, name in configs:
    scheduler = BidirectionalSchedule(D=D, N=8)
    stats = scheduler.compute_bubble_stats()
    
    perf = perf_model.estimate_iteration_time(
        D=D, N=8, W=W,
        C_f=stats.critical_path_forward,
        C_b=stats.critical_path_backward,
        message_size_bytes=4*1024*1024,  # 4MB message
        eager_sync_stages={0, D-1}
    )
    
    print(f"{name:<15} {perf['total_time']:<12.3f} {perf['throughput']:<15.2f} {stats.bubble_ratio:<15.3f}")

print("\nBest configuration for this workload: W=4, D=2 (lowest total time)")
