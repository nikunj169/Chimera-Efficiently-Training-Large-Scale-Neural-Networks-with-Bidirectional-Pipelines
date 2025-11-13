"""
Chimera pipeline engine components (Person A's domain)
"""

# Import only what exists in the files
from .partition import StagePartitioner, MemoryEstimate
from .schedule import (
    BidirectionalSchedule,
    ScheduleType,
    ScheduleSlot,
    BubbleStats
)
from .runtime import (
    StageWorker,
    MicroBatchContext
)
from .recompute import (
    ActivationCheckpointing
)

__all__ = [
    # Partition
    'StagePartitioner',
    'MemoryEstimate',
    
    # Schedule
    'BidirectionalSchedule',
    'ScheduleType',
    'ScheduleSlot',
    'BubbleStats',
    
    # Runtime
    'StageWorker',
    'MicroBatchContext',
    
    # Recompute
    'ActivationCheckpointing',
]
