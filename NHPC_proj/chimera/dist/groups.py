"""
Process group initialization for Chimera distributed training.
Manages W×D process mesh and creates communication groups.
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class ProcessGroups:
    """
    Manages process groups for Chimera pipeline parallelism.
    
    W replicas × D stages = P total processes
    Each rank maps to (replica_id, stage_id)
    """
    
    def __init__(self, world_size: int, W: int, D: int):
        """
        Args:
            world_size: Total number of processes (P = W × D)
            W: Number of pipeline replicas
            D: Number of pipeline stages (must be even)
        """
        if world_size != W * D:
            raise ValueError(f"world_size {world_size} != W × D = {W} × {D}")
        
        if D % 2 != 0:
            raise ValueError("D must be even for bidirectional pipelines")
        
        self.world_size = world_size
        self.W = W
        self.D = D
        
        # Process mesh mapping
        self.rank_to_coords: Dict[int, Tuple[int, int]] = {}  # rank -> (w, d)
        self.coords_to_rank: Dict[Tuple[int, int], int] = {}  # (w, d) -> rank
        
        # Process groups for gradient synchronization (per stage)
        self.stage_groups: Dict[int, dist.ProcessGroup] = {}
        
        # P2P neighbor mappings
        self.down_pipeline_neighbors: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self.up_pipeline_neighbors: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        
        self._initialized = False
    
    def initialize(self):
        """Initialize all process groups and mappings"""
        if self._initialized:
            logger.warning("ProcessGroups already initialized")
            return
        
        # Step 1: Create rank-to-coordinate mapping
        self._create_process_mesh()
        
        # Step 2: Create stage-wise process groups for allreduce
        self._create_stage_groups()
        
        # Step 3: Define P2P neighbor mappings
        self._create_p2p_mappings()
        
        self._initialized = True
        logger.info(f"ProcessGroups initialized: W={self.W}, D={self.D}, world_size={self.world_size}")
    
    def _create_process_mesh(self):
        """
        Create W × D process mesh mapping.
        
        Layout: Row-major ordering
        Rank = w * D + d
        where w ∈ [0, W), d ∈ [0, D)
        """
        for rank in range(self.world_size):
            w = rank // self.D  # Replica ID
            d = rank % self.D   # Stage ID
            
            self.rank_to_coords[rank] = (w, d)
            self.coords_to_rank[(w, d)] = rank
        
        logger.info(f"Created process mesh: {self.W} replicas × {self.D} stages")
    
    def _create_stage_groups(self):
        """
        Create process groups for gradient synchronization.
        
        Each stage d has a group containing all replicas:
        stage_group[d] = {rank(w, d) for w in [0, W)}
        """
        for d in range(self.D):
            # Ranks for stage d across all replicas
            stage_ranks = [self.coords_to_rank[(w, d)] for w in range(self.W)]
            
            # Create process group
            group = dist.new_group(ranks=stage_ranks)
            self.stage_groups[d] = group
            
            logger.debug(f"Stage {d} group: ranks {stage_ranks}")
    
    def _create_p2p_mappings(self):
        """
        Create P2P neighbor mappings for down and up pipelines.
        
        Down pipeline: stage0 → stage1 → ... → stageD-1
        Up pipeline: stageD-1 → ... → stage1 → stage0 (reversed)
        """
        for rank in range(self.world_size):
            w, d = self.rank_to_coords[rank]
            
            # Down pipeline neighbors (+1 stage)
            prev_stage = d - 1 if d > 0 else None
            next_stage = d + 1 if d < self.D - 1 else None
            
            prev_rank = self.coords_to_rank.get((w, prev_stage)) if prev_stage is not None else None
            next_rank = self.coords_to_rank.get((w, next_stage)) if next_stage is not None else None
            
            self.down_pipeline_neighbors[rank] = (prev_rank, next_rank)
            
            # Up pipeline neighbors (reversed mapping)
            # For up pipeline, stages are reversed: D-1 → D-2 → ... → 0
            # So next in up = prev in down
            up_prev = next_rank  # Next stage in down = previous in up
            up_next = prev_rank  # Prev stage in down = next in up
            
            self.up_pipeline_neighbors[rank] = (up_prev, up_next)
        
        logger.info("Created P2P neighbor mappings for down and up pipelines")
    
    def get_coords(self, rank: int) -> Tuple[int, int]:
        """Get (replica_id, stage_id) for a rank"""
        return self.rank_to_coords[rank]
    
    def get_rank(self, replica_id: int, stage_id: int) -> int:
        """Get rank for (replica_id, stage_id)"""
        return self.coords_to_rank[(replica_id, stage_id)]
    
    def get_stage_group(self, stage_id: int) -> dist.ProcessGroup:
        """Get process group for gradient synchronization at a stage"""
        return self.stage_groups[stage_id]
    
    def get_down_neighbors(self, rank: int) -> Tuple[Optional[int], Optional[int]]:
        """Get (previous_rank, next_rank) for down pipeline"""
        return self.down_pipeline_neighbors[rank]
    
    def get_up_neighbors(self, rank: int) -> Tuple[Optional[int], Optional[int]]:
        """Get (previous_rank, next_rank) for up pipeline"""
        return self.up_pipeline_neighbors[rank]
    
    def is_edge_stage(self, rank: int) -> bool:
        """Check if rank is at edge stage (stage 0 or stage D-1)"""
        _, stage_id = self.rank_to_coords[rank]
        return stage_id == 0 or stage_id == self.D - 1
    
    def get_replica_id(self, rank: int) -> int:
        """Get replica ID for a rank"""
        return self.rank_to_coords[rank][0]
    
    def get_stage_id(self, rank: int) -> int:
        """Get stage ID for a rank"""
        return self.rank_to_coords[rank][1]


# Global instance
_process_groups: Optional[ProcessGroups] = None


def init_process_groups(world_size: int, W: int, D: int) -> ProcessGroups:
    """
    Initialize global process groups.
    
    Args:
        world_size: Total number of processes
        W: Number of replicas
        D: Number of stages
    
    Returns:
        ProcessGroups instance
    """
    global _process_groups
    
    if _process_groups is not None:
        logger.warning("Process groups already initialized")
        return _process_groups
    
    _process_groups = ProcessGroups(world_size, W, D)
    _process_groups.initialize()
    
    return _process_groups


def get_process_groups() -> ProcessGroups:
    """Get global ProcessGroups instance"""
    if _process_groups is None:
        raise RuntimeError("Process groups not initialized. Call init_process_groups() first.")
    
    return _process_groups


# Example usage
if __name__ == "__main__":
    # Mock distributed setup
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize for testing (single process)
    dist.init_process_group(backend='gloo', rank=0, world_size=8)
    
    # Create process groups: W=2 replicas, D=4 stages
    pg = init_process_groups(world_size=8, W=2, D=4)
    
    # Test mappings
    print(f"Rank 0 coords: {pg.get_coords(0)}")  # (0, 0)
    print(f"Rank 5 coords: {pg.get_coords(5)}")  # (1, 1)
    
    print(f"Down neighbors of rank 0: {pg.get_down_neighbors(0)}")  # (None, 1)
    print(f"Up neighbors of rank 0: {pg.get_up_neighbors(0)}")      # (1, None)
    
    print(f"Is rank 0 edge stage: {pg.is_edge_stage(0)}")  # True
    print(f"Is rank 1 edge stage: {pg.is_edge_stage(1)}")  # False
    
    dist.destroy_process_group()
