"""
End-to-end training orchestration for Chimera.
Handles distributed setup, data loading, and training loop.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, Optional

from chimera.engine import BidirectionalSchedule, ScheduleType, StageWorker
from chimera.dist import init_process_groups, P2PHandler, AllReduceHandler
from chimera.models import BertForPipelineParallelism, BertConfig
from chimera.config import PerformanceModel, AutoTuner


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChimeraTrainer:
    """
    End-to-end trainer for Chimera pipeline parallelism.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Training configuration dict
        """
        self.config = config
        
        # Distributed setup
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Pipeline configuration
        self.W = config['W']  # Replicas
        self.D = config['D']  # Stages
        self.N = config['N']  # Micro-batches
        self.B = config['B']  # Micro-batch size
        
        # Initialize process groups
        self.process_groups = init_process_groups(self.world_size, self.W, self.D)
        self.replica_id = self.process_groups.get_replica_id(self.rank)
        self.stage_id = self.process_groups.get_stage_id(self.rank)
        
        logger.info(f"Rank {self.rank}: Replica {self.replica_id}, Stage {self.stage_id}")
        
        # Create model stage
        self.model_stage = self._create_model_stage()
        
        # Create schedule
        self.schedule = self._create_schedule()
        
        # Communication handlers
        self.p2p = P2PHandler(self.rank, self.process_groups)
        
        eager_stages = self.schedule.get_eager_sync_stages()
        self.allreduce = AllReduceHandler(self.rank, self.process_groups, eager_stages)
        self.allreduce.register_gradients(self.model_stage)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model_stage.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        
        # Metrics
        self.metrics = {
            'iteration_times': [],
            'throughput': [],
            'loss': []
        }
    
    def _create_model_stage(self) -> nn.Module:
        """Create model stage for this rank"""
        model_config = self.config['model']
        
        if model_config['type'] == 'bert':
            config = BertConfig(**model_config['config'])
            full_model = BertForPipelineParallelism(config, num_stages=self.D)
            stage = full_model.get_stage(self.stage_id)
        else:
            raise ValueError(f"Unknown model type: {model_config['type']}")
        
        if torch.cuda.is_available():
            stage = stage.cuda()
        
        return stage
    
    def _create_schedule(self) -> BidirectionalSchedule:
        """Create pipeline schedule"""
        scheduler = BidirectionalSchedule(D=self.D, N=self.N, W=self.W)
        
        strategy_name = self.config.get('schedule_strategy', 'BASE')
        strategy = ScheduleType[strategy_name]
        
        schedule = scheduler.build_schedule(strategy)
        
        # Get schedule for this rank
        self.rank_schedule = scheduler.get_schedule_for_rank(self.rank)
        
        logger.info(f"Rank {self.rank}: {len(self.rank_schedule)} operations in schedule")
        
        return scheduler
    
    def train_iteration(self) -> Dict[str, float]:
        """
        Execute one training iteration.
        
        Returns:
            Dict with iteration metrics
        """
        start_time = time.perf_counter()
        
        # Execute scheduled operations
        total_loss = 0.0
        loss_count = 0
        
        for slot in self.rank_schedule:
            if slot.operation == 'forward':
                self._forward_step(slot)
            elif slot.operation == 'backward':
                loss = self._backward_step(slot)
                if loss is not None:
                    total_loss += loss
                    loss_count += 1
        
        # Synchronize gradients
        self.allreduce.sync_all_gradients(self.model_stage)
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(self.model_stage.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Global barrier
        dist.barrier()
        
        end_time = time.perf_counter()
        iteration_time = end_time - start_time
        
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0
        
        metrics = {
            'iteration_time': iteration_time,
            'throughput': self.N / iteration_time,
            'loss': avg_loss
        }
        
        return metrics
    
    def _forward_step(self, slot):
        """Execute forward pass for a slot"""
        # Simplified: actual implementation would integrate with StageWorker
        pass
    
    def _backward_step(self, slot) -> Optional[float]:
        """Execute backward pass for a slot"""
        # Simplified: actual implementation would compute loss and backward
        return None
    
    def train(self, num_iterations: int):
        """
        Run training for specified iterations.
        
        Args:
            num_iterations: Number of training iterations
        """
        logger.info(f"Rank {self.rank}: Starting training for {num_iterations} iterations")
        
        for iteration in range(num_iterations):
            metrics = self.train_iteration()
            
            # Log metrics
            if self.rank == 0 and iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"Time={metrics['iteration_time']:.3f}s, "
                    f"Throughput={metrics['throughput']:.2f} mb/s, "
                    f"Loss={metrics['loss']:.4f}"
                )
            
            # Store metrics
            self.metrics['iteration_times'].append(metrics['iteration_time'])
            self.metrics['throughput'].append(metrics['throughput'])
            self.metrics['loss'].append(metrics['loss'])
        
        # Report final statistics
        if self.rank == 0:
            avg_time = sum(self.metrics['iteration_times']) / len(self.metrics['iteration_times'])
            avg_throughput = sum(self.metrics['throughput']) / len(self.metrics['throughput'])
            
            logger.info(f"\nTraining completed!")
            logger.info(f"  Average iteration time: {avg_time:.3f}s")
            logger.info(f"  Average throughput: {avg_throughput:.2f} micro-batches/s")
    
    def save_metrics(self, output_path: str):
        """Save metrics to file"""
        if self.rank == 0:
            import json
            
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            logger.info(f"Metrics saved to {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Chimera Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--num-iterations', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    
    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize distributed
    dist.init_process_group(backend='gloo')
    
    # Create trainer
    trainer = ChimeraTrainer(config)
    
    # Train
    trainer.train(args.num_iterations)
    
    # Save metrics
    output_path = Path(args.output_dir) / f'metrics_rank{trainer.rank}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_metrics(str(output_path))
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
