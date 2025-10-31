"""Demonstrate Chimera schedule generation and visualization"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from chimera.engine import BidirectionalSchedule

print("Chimera Schedule Demonstration")
print("="*60)

# Create schedule for D=4 stages, N=8 micro-batches
scheduler = BidirectionalSchedule(D=4, N=8)
schedule = scheduler.build_schedule()

# Visualize
print("\nSchedule Timeline:")
print(scheduler.visualize_schedule())

# Show statistics
stats = scheduler.compute_bubble_stats()
print(f"\nSchedule Statistics:")
print(f"  Pipeline depth (D): {scheduler.D}")
print(f"  Micro-batches (N): {scheduler.N}")
print(f"  Total bubbles: {stats.total_bubbles}")
print(f"  Bubble ratio: {stats.bubble_ratio:.3f}")
print(f"  Critical path (Cf): {stats.critical_path_forward}")
print(f"  Critical path (Cb): {stats.critical_path_backward}")

# Show eager sync stages
eager_stages = scheduler.get_eager_sync_stages()
print(f"  Eager sync stages: {eager_stages}")

# Compare with GPipe baseline
gpipe_bubbles = 2 * (scheduler.D - 1)
chimera_bubbles = scheduler.D - 2
improvement = (gpipe_bubbles - chimera_bubbles) / gpipe_bubbles * 100

print(f"\nBubble Reduction vs GPipe:")
print(f"  GPipe bubbles: {gpipe_bubbles}")
print(f"  Chimera bubbles: {chimera_bubbles}")
print(f"  Improvement: {improvement:.1f}%")

print("\n" + "="*60)
print("✅ Schedule demonstration complete!")
