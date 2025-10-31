"""
Point-to-point communication for activation and gradient transfer.
Supports nonblocking operations with backpressure handling.
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple
import logging
from collections import deque


logger = logging.getLogger(__name__)


class P2PHandler:
    """
    Handles point-to-point communication between pipeline stages.
    Sends activations forward and gradients backward.
    """
    
    def __init__(self, rank: int, process_groups, backend: str = 'gloo'):
        """
        Args:
            rank: Current process rank
            process_groups: ProcessGroups instance
            backend: Communication backend ('gloo' or 'nccl')
        """
        self.rank = rank
        self.pg = process_groups
        self.backend = backend
        
        # Pending operations for nonblocking send/recv
        self.pending_sends = deque()
        self.pending_recvs = deque()
        
        # Timeout for operations (seconds)
        self.timeout_seconds = 300.0
        
        logger.info(f"P2PHandler initialized for rank {self.rank} with backend {backend}")
    
    def send_activation(
        self,
        tensor: torch.Tensor,
        dst_stage: int,
        micro_batch_id: int,
        blocking: bool = True
    ) -> Optional[dist.Work]:
        """
        Send activation tensor to next stage.
        
        Args:
            tensor: Activation tensor to send
            dst_stage: Destination stage ID
            micro_batch_id: Micro-batch identifier
            blocking: If True, wait for completion
        
        Returns:
            dist.Work handle if nonblocking, None if blocking
        """
        replica_id = self.pg.get_replica_id(self.rank)
        dst_rank = self.pg.get_rank(replica_id, dst_stage)
        
        logger.debug(f"Rank {self.rank} sending activation mb={micro_batch_id} to rank {dst_rank}")
        
        # Send tensor metadata first (shape, dtype)
        metadata = self._create_metadata(tensor, micro_batch_id)
        
        # Blocking send for metadata
        dist.send(metadata, dst=dst_rank)
        
        # Send actual tensor
        if blocking:
            dist.send(tensor.contiguous(), dst=dst_rank)
            return None
        else:
            # Nonblocking send
            work = dist.isend(tensor.contiguous(), dst=dst_rank)
            self.pending_sends.append((work, tensor, micro_batch_id))
            return work
    
    def recv_activation(
        self,
        src_stage: int,
        micro_batch_id: int,
        blocking: bool = True
    ) -> torch.Tensor:
        """
        Receive activation tensor from previous stage.
        
        Args:
            src_stage: Source stage ID
            micro_batch_id: Micro-batch identifier
            blocking: If True, wait for completion
        
        Returns:
            Received activation tensor
        """
        replica_id = self.pg.get_replica_id(self.rank)
        src_rank = self.pg.get_rank(replica_id, src_stage)
        
        logger.debug(f"Rank {self.rank} receiving activation mb={micro_batch_id} from rank {src_rank}")
        
        # Receive metadata first
        metadata = torch.zeros(10, dtype=torch.long)
        dist.recv(metadata, src=src_rank)
        
        shape, dtype_code = self._parse_metadata(metadata)
        
        # Allocate tensor
        tensor = torch.zeros(shape, dtype=self._code_to_dtype(dtype_code))
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Receive actual tensor
        if blocking:
            dist.recv(tensor, src=src_rank)
        else:
            # Nonblocking recv
            work = dist.irecv(tensor, src=src_rank)
            self.pending_recvs.append((work, tensor, micro_batch_id))
            work.wait()  # For simplicity, wait immediately
        
        return tensor
    
    def send_gradient(
        self,
        tensor: torch.Tensor,
        dst_stage: int,
        micro_batch_id: int,
        blocking: bool = True
    ) -> Optional[dist.Work]:
        """
        Send gradient tensor to previous stage.
        
        Args:
            tensor: Gradient tensor to send
            dst_stage: Destination stage ID
            micro_batch_id: Micro-batch identifier
            blocking: If True, wait for completion
        
        Returns:
            dist.Work handle if nonblocking, None if blocking
        """
        replica_id = self.pg.get_replica_id(self.rank)
        dst_rank = self.pg.get_rank(replica_id, dst_stage)
        
        logger.debug(f"Rank {self.rank} sending gradient mb={micro_batch_id} to rank {dst_rank}")
        
        # Send metadata
        metadata = self._create_metadata(tensor, micro_batch_id)
        dist.send(metadata, dst=dst_rank)
        
        # Send gradient
        if blocking:
            dist.send(tensor.contiguous(), dst=dst_rank)
            return None
        else:
            work = dist.isend(tensor.contiguous(), dst=dst_rank)
            self.pending_sends.append((work, tensor, micro_batch_id))
            return work
    
    def recv_gradient(
        self,
        src_stage: int,
        micro_batch_id: int,
        blocking: bool = True
    ) -> torch.Tensor:
        """
        Receive gradient tensor from next stage.
        
        Args:
            src_stage: Source stage ID
            micro_batch_id: Micro-batch identifier
            blocking: If True, wait for completion
        
        Returns:
            Received gradient tensor
        """
        replica_id = self.pg.get_replica_id(self.rank)
        src_rank = self.pg.get_rank(replica_id, src_stage)
        
        logger.debug(f"Rank {self.rank} receiving gradient mb={micro_batch_id} from rank {src_rank}")
        
        # Receive metadata
        metadata = torch.zeros(10, dtype=torch.long)
        dist.recv(metadata, src=src_rank)
        
        shape, dtype_code = self._parse_metadata(metadata)
        
        # Allocate tensor
        tensor = torch.zeros(shape, dtype=self._code_to_dtype(dtype_code))
        
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Receive gradient
        if blocking:
            dist.recv(tensor, src=src_rank)
        else:
            work = dist.irecv(tensor, src=src_rank)
            self.pending_recvs.append((work, tensor, micro_batch_id))
            work.wait()
        
        return tensor
    
    def wait_all_sends(self):
        """Wait for all pending send operations to complete"""
        while self.pending_sends:
            work, tensor, mb_id = self.pending_sends.popleft()
            work.wait()
            logger.debug(f"Rank {self.rank} completed send for mb={mb_id}")
    
    def wait_all_recvs(self):
        """Wait for all pending receive operations to complete"""
        while self.pending_recvs:
            work, tensor, mb_id = self.pending_recvs.popleft()
            work.wait()
            logger.debug(f"Rank {self.rank} completed recv for mb={mb_id}")
    
    def _create_metadata(self, tensor: torch.Tensor, micro_batch_id: int) -> torch.Tensor:
        """
        Create metadata tensor with shape and dtype info.
        
        Format: [mb_id, ndim, dim0, dim1, ..., dtype_code, padding...]
        """
        metadata = torch.zeros(10, dtype=torch.long)
        metadata[0] = micro_batch_id
        metadata[1] = len(tensor.shape)
        
        for i, dim in enumerate(tensor.shape):
            if i < 6:  # Max 6 dimensions
                metadata[2 + i] = dim
        
        metadata[8] = self._dtype_to_code(tensor.dtype)
        
        return metadata
    
    def _parse_metadata(self, metadata: torch.Tensor) -> Tuple[tuple, int]:
        """Parse metadata tensor to extract shape and dtype"""
        ndim = int(metadata[1].item())
        shape = tuple(int(metadata[2 + i].item()) for i in range(ndim))
        dtype_code = int(metadata[8].item())
        
        return shape, dtype_code
    
    def _dtype_to_code(self, dtype: torch.dtype) -> int:
        """Convert dtype to integer code"""
        dtype_map = {
            torch.float32: 0,
            torch.float16: 1,
            torch.bfloat16: 2,
            torch.int64: 3,
            torch.int32: 4,
        }
        return dtype_map.get(dtype, 0)
    
    def _code_to_dtype(self, code: int) -> torch.dtype:
        """Convert integer code to dtype"""
        code_map = {
            0: torch.float32,
            1: torch.float16,
            2: torch.bfloat16,
            3: torch.int64,
            4: torch.int32,
        }
        return code_map.get(code, torch.float32)


# Example usage
if __name__ == "__main__":
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    # Initialize distributed
    dist.init_process_group(backend='gloo', rank=0, world_size=2)
    
    from chimera.dist.groups import init_process_groups
    
    # Create process groups
    pg = init_process_groups(world_size=2, W=1, D=2)
    
    # Create P2P handler
    p2p = P2PHandler(rank=0, process_groups=pg)
    
    # Test (would need rank 1 to actually send/recv)
    print(f"P2P handler created for rank {p2p.rank}")
    
    dist.destroy_process_group()
