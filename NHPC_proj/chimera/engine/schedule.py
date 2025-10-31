"""
Bidirectional pipeline scheduling for Chimera.
Generates conflict-free merged schedules with 1F1B strategy.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ScheduleType(Enum):
    """N>D scheduling strategies"""
    BASE = "base"
    DIRECT_CONCAT = "direct_concatenation"
    FORWARD_DOUBLING = "forward_doubling"
    BACKWARD_HALVING = "backward_halving"


@dataclass
class ScheduleSlot:
    """Single operation in pipeline schedule"""
    time: int
    stage_id: int
    micro_batch_id: int
    operation: str
    requires_recompute: bool = False
    doubled_forward: bool = False


@dataclass
class BubbleStats:
    """Pipeline bubble statistics"""
    forward_bubbles: int
    backward_bubbles: int
    total_bubbles: int
    bubble_ratio: float
    critical_path_forward: int
    critical_path_backward: int


class BidirectionalSchedule:
    """
    Chimera bidirectional pipeline scheduler.
    Merges down and up pipelines with 1F1B strategy.
    """
    
    def __init__(self, D: int, N: int, W: int = 1):
        """
        Args:
            D: Number of pipeline stages (must be even)
            N: Number of micro-batches per iteration
            W: Number of replicated pipelines (data parallelism)
        """
        if D % 2 != 0:
            raise ValueError("D must be even for bidirectional pipelines")
        if N < 1:
            raise ValueError("N must be at least 1")
        
        self.D = D
        self.N = N
        self.W = W
        self.schedule: Dict[int, List[ScheduleSlot]] = {}
        
    def build_schedule(self, strategy: ScheduleType = ScheduleType.BASE) -> Dict[int, List[ScheduleSlot]]:
        """Build complete schedule based on strategy"""
        if strategy == ScheduleType.BASE or self.N <= self.D:
            return self.build_base_schedule()
        elif strategy == ScheduleType.DIRECT_CONCAT:
            return self.build_direct_concatenation()
        elif strategy == ScheduleType.FORWARD_DOUBLING:
            return self.build_forward_doubling()
        elif strategy == ScheduleType.BACKWARD_HALVING:
            return self.build_backward_halving()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def build_base_schedule(self) -> Dict[int, List[ScheduleSlot]]:
        """Base schedule for N = D or N < D"""
        if self.N < self.D:
            n_down = self.N // 2
            n_up = self.N - n_down
        else:
            n_down = self.N // 2
            n_up = self.N // 2
        
        down_batches = list(range(0, n_down))
        up_batches = list(range(n_down, self.N))
        
        # Build pipelines
        down_pipeline = self._build_1F1B_pipeline(
            micro_batches=down_batches,
            stage_mapping=list(range(self.D)),
            direction="down"
        )
        
        up_pipeline = self._build_1F1B_pipeline(
            micro_batches=up_batches,
            stage_mapping=list(reversed(range(self.D))),
            direction="up"
        )
        
        self.schedule = self._merge_pipelines(down_pipeline, up_pipeline)
        return self.schedule
    
    def _build_1F1B_pipeline(self, micro_batches: List[int],
                            stage_mapping: List[int],
                            direction: str) -> Dict[int, List[ScheduleSlot]]:
        """
        Build 1F1B pipeline with correct timing to avoid conflicts.
        """
        timeline = {rank: [] for rank in range(self.D)}
        N = len(micro_batches)
        
        if N == 0:
            return timeline
        
        current_time = 0
        
        # Phase 1: Warm-up - Fill pipeline with forwards
        for i in range(min(N, self.D)):
            mb_id = micro_batches[i]
            
            for stage_id in range(self.D):
                worker = stage_mapping[stage_id]
                timeline[worker].append(ScheduleSlot(
                    time=current_time,
                    stage_id=stage_id,
                    micro_batch_id=mb_id,
                    operation='forward'
                ))
                current_time += 1
        
        # Phase 2: Steady-state - 1F1B
        for i in range(self.D, N):
            mb_id = micro_batches[i]
            
            # Forward for new micro-batch
            for stage_id in range(self.D):
                worker = stage_mapping[stage_id]
                timeline[worker].append(ScheduleSlot(
                    time=current_time,
                    stage_id=stage_id,
                    micro_batch_id=mb_id,
                    operation='forward'
                ))
                current_time += 1
            
            # Backward for completed micro-batch
            oldest_mb = micro_batches[i - self.D]
            for stage_id in reversed(range(self.D)):
                worker = stage_mapping[stage_id]
                timeline[worker].append(ScheduleSlot(
                    time=current_time,
                    stage_id=stage_id,
                    micro_batch_id=oldest_mb,
                    operation='backward'
                ))
                current_time += 1
        
        # Phase 3: Cool-down - Drain remaining backwards
        for i in range(max(0, N - self.D), N):
            mb_id = micro_batches[i]
            
            for stage_id in reversed(range(self.D)):
                worker = stage_mapping[stage_id]
                timeline[worker].append(ScheduleSlot(
                    time=current_time,
                    stage_id=stage_id,
                    micro_batch_id=mb_id,
                    operation='backward'
                ))
                current_time += 1
        
        return timeline
    
    def _merge_pipelines(self, down: Dict[int, List[ScheduleSlot]],
                        up: Dict[int, List[ScheduleSlot]]) -> Dict[int, List[ScheduleSlot]]:
        """Merge down and up pipeline schedules"""
        merged = {}
        
        # Find max time in down pipeline to offset up pipeline
        max_time_down = 0
        for slots in down.values():
            if slots:
                max_time_down = max(max_time_down, max(s.time for s in slots))
        
        # Offset up pipeline to start after down pipeline
        time_offset = max_time_down + 1
        
        for rank in range(self.D):
            merged[rank] = down[rank].copy()
            
            # Add up pipeline slots with time offset
            for slot in up[rank]:
                new_slot = ScheduleSlot(
                    time=slot.time + time_offset,
                    stage_id=slot.stage_id,
                    micro_batch_id=slot.micro_batch_id,
                    operation=slot.operation,
                    requires_recompute=slot.requires_recompute,
                    doubled_forward=slot.doubled_forward
                )
                merged[rank].append(new_slot)
            
            # Sort by time
            merged[rank] = sorted(merged[rank], key=lambda s: s.time)
        
        # Validate no conflicts
        self._validate_no_conflicts(merged)
        
        return merged
    
    def _validate_no_conflicts(self, schedule: Dict[int, List[ScheduleSlot]]):
        """Verify no two operations occur at same time on same worker"""
        for rank, slots in schedule.items():
            time_map = {}
            for slot in slots:
                if slot.time in time_map:
                    raise RuntimeError(
                        f"Conflict detected at rank {rank}, time {slot.time}: "
                        f"{time_map[slot.time]} and {slot}"
                    )
                time_map[slot.time] = slot
    
    def build_direct_concatenation(self) -> Dict[int, List[ScheduleSlot]]:
        """Direct concatenation for N > D"""
        K = self.N // self.D
        residual = self.N % self.D
        
        full_schedule = {rank: [] for rank in range(self.D)}
        time_offset = 0
        
        for k in range(K):
            unit_schedule = self._build_base_unit_schedule(
                start_mb_id=k * self.D,
                num_batches=self.D
            )
            
            for rank in range(self.D):
                for slot in unit_schedule[rank]:
                    new_slot = ScheduleSlot(
                        time=slot.time + time_offset,
                        stage_id=slot.stage_id,
                        micro_batch_id=slot.micro_batch_id,
                        operation=slot.operation
                    )
                    full_schedule[rank].append(new_slot)
            
            max_time = max(slot.time for slots in unit_schedule.values() for slot in slots)
            time_offset += max_time + 1
        
        if residual > 0:
            residual_schedule = self._build_base_unit_schedule(
                start_mb_id=K * self.D,
                num_batches=residual
            )
            
            for rank in range(self.D):
                for slot in residual_schedule[rank]:
                    new_slot = ScheduleSlot(
                        time=slot.time + time_offset,
                        stage_id=slot.stage_id,
                        micro_batch_id=slot.micro_batch_id,
                        operation=slot.operation
                    )
                    full_schedule[rank].append(new_slot)
        
        self.schedule = full_schedule
        return full_schedule
    
    def _build_base_unit_schedule(self, start_mb_id: int,
                                  num_batches: int) -> Dict[int, List[ScheduleSlot]]:
        """Helper to build a single base unit schedule"""
        temp_scheduler = BidirectionalSchedule(self.D, num_batches, self.W)
        unit_schedule = temp_scheduler.build_base_schedule()
        
        for rank in range(self.D):
            for slot in unit_schedule[rank]:
                slot.micro_batch_id += start_mb_id
        
        return unit_schedule
    
    def build_forward_doubling(self) -> Dict[int, List[ScheduleSlot]]:
        """Forward doubling for N > D"""
        if self.N % (2 * self.D) != 0:
            raise ValueError("N must be multiple of 2D for forward doubling")
        
        schedule = {rank: [] for rank in range(self.D)}
        current_time = 0
        num_forward_slots = self.N // 2
        
        # Forward phase
        for slot_idx in range(num_forward_slots):
            mb1 = 2 * slot_idx
            mb2 = 2 * slot_idx + 1
            
            for stage_id in range(self.D):
                for mb_id in [mb1, mb2]:
                    schedule[stage_id].append(ScheduleSlot(
                        time=current_time,
                        stage_id=stage_id,
                        micro_batch_id=mb_id,
                        operation='forward',
                        doubled_forward=True
                    ))
                current_time += 1
        
        # Backward phase
        for mb_id in range(self.N):
            for stage_id in reversed(range(self.D)):
                schedule[stage_id].append(ScheduleSlot(
                    time=current_time,
                    stage_id=stage_id,
                    micro_batch_id=mb_id,
                    operation='backward',
                    requires_recompute=True
                ))
                current_time += 1
        
        self.schedule = schedule
        return schedule
    
    def build_backward_halving(self) -> Dict[int, List[ScheduleSlot]]:
        """Backward halving for N > D"""
        schedule = self.build_forward_doubling()
        
        for rank in range(self.D):
            for slot in schedule[rank]:
                if slot.operation == 'backward':
                    slot.requires_recompute = False
        
        self.schedule = schedule
        return schedule
    
    def compute_bubble_stats(self) -> BubbleStats:
        """Compute pipeline bubble statistics"""
        Cf = 2 * self.N + (self.D - 2)
        Cb = 2 * self.N + (self.D - 2)
        
        forward_bubbles = self.D // 2 - 1
        backward_bubbles = self.D // 2 - 1
        total_bubbles = self.D - 2
        
        bubble_ratio = total_bubbles / (2 * self.N + self.D - 2)
        
        return BubbleStats(
            forward_bubbles=forward_bubbles,
            backward_bubbles=backward_bubbles,
            total_bubbles=total_bubbles,
            bubble_ratio=bubble_ratio,
            critical_path_forward=Cf,
            critical_path_backward=Cb
        )
    
    def get_eager_sync_stages(self) -> set:
        """Return stage IDs eligible for eager gradient synchronization"""
        return {0, self.D - 1}
    
    def get_schedule_for_rank(self, rank: int) -> List[ScheduleSlot]:
        """Get schedule timeline for specific worker rank"""
        if rank not in self.schedule:
            raise ValueError(f"No schedule for rank {rank}")
        return self.schedule[rank]
    
    def visualize_schedule(self, max_time: Optional[int] = None) -> str:
        """ASCII visualization of schedule timeline"""
        if not self.schedule:
            return "No schedule built"
        
        all_times = [slot.time for slots in self.schedule.values() for slot in slots]
        max_t = max(all_times) if not max_time else max_time
        
        viz = []
        viz.append(f"Chimera Schedule (D={self.D}, N={self.N})")
        viz.append("=" * 60)
        
        for rank in range(self.D):
            line = f"Rank {rank}: "
            slots_by_time = {slot.time: slot for slot in self.schedule[rank]}
            
            for t in range(min(max_t + 1, 30)):  # Limit display
                if t in slots_by_time:
                    slot = slots_by_time[t]
                    symbol = 'F' if slot.operation == 'forward' else 'B'
                    line += f"{symbol}{slot.micro_batch_id} "
                else:
                    line += "__ "
            
            viz.append(line)
        
        return "\n".join(viz)


if __name__ == "__main__":
    # Test
    scheduler = BidirectionalSchedule(D=4, N=4)
    schedule = scheduler.build_schedule()
    print(scheduler.visualize_schedule())
    print()
    stats = scheduler.compute_bubble_stats()
    print(f"Bubbles: {stats.total_bubbles}, Ratio: {stats.bubble_ratio:.3f}")
