"""
Model Checkpointing System for Bayesian Expectation Transformer

Implements a hierarchical 3-tier checkpointing strategy:
1. Training Checkpoints - Frequent, rolling retention
2. Milestone Checkpoints - On improvement, indefinite retention
3. Production Checkpoints - Versioned, minimal storage

References:
- ADR-004: Model Checkpointing Strategy
- System Architecture: Section 3.3 Model Checkpointing System
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json
import socket
import sys
import subprocess
import shutil


class CheckpointManager:
    """Hierarchical checkpoint manager with 3 tiers.

    Checkpoint Types:
    - Training: Frequent, rolling (keeps last N)
    - Milestone: On improvement, indefinite retention
    - Production: Versioned, minimal storage

    Features:
    - Atomic writes (no corruption)
    - Rolling retention for training checkpoints
    - Full state preservation for training resume
    - Versioned production checkpoints
    - Metadata tracking (git commit, hardware, etc.)

    Example:
        >>> manager = CheckpointManager(checkpoint_dir='checkpoints')
        >>>
        >>> # During training
        >>> manager.save_training_checkpoint(
        ...     model, optimizer, epoch=5, step=1000,
        ...     metrics={'loss': 0.25, 'accuracy': 0.85}
        ... )
        >>>
        >>> # On validation improvement
        >>> manager.save_milestone_checkpoint(
        ...     model, metric_value=0.87, metric_name='accuracy',
        ...     epoch=7, metadata={'notes': 'best so far'}
        ... )
        >>>
        >>> # For production deployment
        >>> manager.save_production_checkpoint(
        ...     model, version='1.0.0', config=config,
        ...     metrics={'accuracy': 0.89}
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        max_training_checkpoints: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        compress: bool = False
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Root directory for checkpoints
            max_training_checkpoints: Maximum training checkpoints to keep
            save_optimizer: Whether to save optimizer state (needed for resume)
            save_scheduler: Whether to save scheduler state
            compress: Whether to compress checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.training_dir = self.checkpoint_dir / 'training'
        self.milestone_dir = self.checkpoint_dir / 'milestone'
        self.production_dir = self.checkpoint_dir / 'production'
        self.metadata_dir = self.checkpoint_dir / 'metadata'

        # Create directory structure
        for dir_path in [self.training_dir, self.milestone_dir,
                         self.production_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.max_training_checkpoints = max_training_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.compress = compress
        self.best_metric = float('-inf')

        print(f"CheckpointManager initialized at: {self.checkpoint_dir.absolute()}")

    def save_training_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Any] = None
    ) -> str:
        """Save training checkpoint with rolling retention.

        Includes full training state for exact resume:
        - Model weights
        - Optimizer state (momentum, etc.)
        - Scheduler state
        - Training progress (epoch, step)
        - Metrics history
        - Configuration
        - Metadata (hardware, git commit, etc.)

        Args:
            model: PyTorch model to save
            optimizer: Optimizer instance (optional but recommended)
            epoch: Current epoch number
            step: Current global step
            metrics: Dictionary of current metrics
            config: Model/training configuration
            scheduler: Learning rate scheduler (optional)

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            'version': '1.0.0',
            'checkpoint_type': 'training',
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),

            # Model state
            'model_state_dict': model.state_dict(),

            # Training state
            'epoch': epoch,
            'global_step': step,
            'metrics': metrics,

            # Configuration
            'config': config or {},

            # Environment metadata
            'metadata': self._get_metadata()
        }

        # Add optimizer state if requested
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        # Add scheduler state if requested
        if self.save_scheduler and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Atomic write to prevent corruption
        filename = f'checkpoint_epoch_{epoch:04d}_step_{step:08d}.pt'
        filepath = self.training_dir / filename
        temp_path = filepath.with_suffix('.tmp')

        # Save to temporary file first
        torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=self.compress)

        # Atomic rename (prevents corruption if interrupted)
        temp_path.replace(filepath)

        # Update 'latest' symlink
        self._update_symlink(self.training_dir / 'latest.pt', filename)

        # Rolling cleanup - keep only last N checkpoints
        self._cleanup_old_training_checkpoints()

        print(f"Training checkpoint saved: {filepath.name}")
        return str(filepath)

    def save_milestone_checkpoint(
        self,
        model: torch.nn.Module,
        metric_value: float,
        metric_name: str,
        epoch: int,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None
    ) -> Optional[str]:
        """Save milestone checkpoint if metric improved.

        Milestone checkpoints are saved indefinitely when the model
        achieves a new best performance on a tracked metric.

        Args:
            model: PyTorch model to save
            metric_value: Current metric value
            metric_name: Name of metric (e.g., 'accuracy', 'loss')
            epoch: Current epoch number
            config: Model configuration
            metadata: Additional metadata
            optimizer: Optimizer state (optional, for full resume)
            scheduler: Scheduler state (optional)

        Returns:
            Path to saved checkpoint if improved, None otherwise
        """
        # Check if metric improved (handle both maximization and minimization)
        is_loss_metric = 'loss' in metric_name.lower()
        improved = (metric_value < self.best_metric if is_loss_metric
                   else metric_value > self.best_metric)

        if not improved and self.best_metric != float('-inf'):
            print(f"Metric {metric_name}={metric_value:.4f} did not improve "
                  f"(best: {self.best_metric:.4f})")
            return None

        # Update best metric
        self.best_metric = metric_value

        checkpoint = {
            'version': '1.0.0',
            'checkpoint_type': 'milestone',
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),

            # Model state
            'model_state_dict': model.state_dict(),

            # Milestone information
            'epoch': epoch,
            'metric_name': metric_name,
            'metric_value': metric_value,

            # Configuration
            'config': config or {},

            # Custom metadata
            'metadata': {
                **(metadata or {}),
                **self._get_metadata()
            }
        }

        # Optionally include optimizer/scheduler for full resume
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Create descriptive filename
        filename = f'best_{metric_name}_{metric_value:.4f}_epoch_{epoch:04d}.pt'
        filepath = self.milestone_dir / filename

        # Save checkpoint
        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=self.compress)

        # Save metadata JSON separately for easy inspection
        meta_file = filepath.with_suffix('.json')
        with open(meta_file, 'w') as f:
            json.dump({
                'metric': metric_value,
                'metric_name': metric_name,
                'epoch': epoch,
                'timestamp': checkpoint['timestamp'],
                'config': config or {}
            }, f, indent=2)

        # Update best_{metric_name}.pt symlink
        self._update_symlink(
            self.milestone_dir / f'best_{metric_name}.pt',
            filename
        )

        print(f"Milestone checkpoint saved: {filepath.name}")
        print(f"  {metric_name} improved to {metric_value:.4f}")
        return str(filepath)

    def save_production_checkpoint(
        self,
        model: torch.nn.Module,
        version: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        model_card: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save production checkpoint with full versioning.

        Production checkpoints are minimal (no optimizer state) and
        include comprehensive documentation via model cards.

        Args:
            model: PyTorch model to save
            version: Semantic version (e.g., '1.0.0')
            config: Model configuration for inference
            metrics: Final performance metrics
            model_card: Model documentation and metadata

        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            'version': version,
            'checkpoint_type': 'production',
            'timestamp': datetime.now().isoformat(),

            # Model state (inference only)
            'model_state_dict': model.state_dict(),

            # Configuration (inference requirements)
            'config': config,

            # Performance metrics
            'metrics': metrics,

            # Model card information
            'model_card': model_card or {
                'description': 'Bayesian Expectation Transformer',
                'training_date': datetime.now().isoformat(),
                'framework': f'PyTorch {torch.__version__}'
            }
        }

        # Create versioned filename
        filename = f'bayesian_transformer_v{version}.pt'
        filepath = self.production_dir / filename

        # Save with compression for production
        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=True)

        # Save config JSON separately
        config_file = filepath.with_suffix('.config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        # Save model card YAML
        if model_card:
            self._save_model_card(version, model_card, metrics)

        # Update 'latest' symlink
        self._update_symlink(self.production_dir / 'latest.pt', filename)

        print(f"Production checkpoint saved: {filepath.name}")
        print(f"  Version: {version}")
        print(f"  Metrics: {metrics}")
        return str(filepath)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """Load checkpoint and restore state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model instance to load weights into
            optimizer: Optimizer instance to restore (optional)
            scheduler: Scheduler instance to restore (optional)
            device: Device to load checkpoint to

        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint (weights_only=False for backward compatibility with older checkpoints)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Validate checkpoint structure
        required_keys = ['model_state_dict']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint: missing keys {missing_keys}")

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model state loaded from: {checkpoint_path.name}")

        # Load optimizer state if available and requested
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state restored")

        # Load scheduler state if available and requested
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Scheduler state restored")

        # Return metadata for inspection
        metadata = {
            'version': checkpoint.get('version', 'unknown'),
            'checkpoint_type': checkpoint.get('checkpoint_type', 'unknown'),
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }

        if 'timestamp' in checkpoint:
            print(f"  Created: {checkpoint['timestamp']}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")

        return metadata

    def get_latest_checkpoint(self, checkpoint_type: str = 'training') -> Optional[Path]:
        """Get path to latest checkpoint of specified type.

        Args:
            checkpoint_type: Type of checkpoint ('training', 'milestone', 'production')

        Returns:
            Path to latest checkpoint or None if not found
        """
        if checkpoint_type == 'training':
            latest_link = self.training_dir / 'latest.pt'
        elif checkpoint_type == 'milestone':
            # Find most recent milestone
            checkpoints = sorted(
                self.milestone_dir.glob('best_*.pt'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            return checkpoints[0] if checkpoints else None
        elif checkpoint_type == 'production':
            latest_link = self.production_dir / 'latest.pt'
        else:
            raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")

        if latest_link.exists():
            return latest_link
        return None

    def list_checkpoints(self, checkpoint_type: Optional[str] = None) -> Dict[str, list]:
        """List all available checkpoints.

        Args:
            checkpoint_type: Filter by type or None for all

        Returns:
            Dictionary mapping checkpoint types to lists of paths
        """
        results = {}

        types_to_check = ['training', 'milestone', 'production']
        if checkpoint_type:
            types_to_check = [checkpoint_type]

        for ctype in types_to_check:
            if ctype == 'training':
                checkpoints = sorted(
                    self.training_dir.glob('checkpoint_*.pt'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
            elif ctype == 'milestone':
                checkpoints = sorted(
                    self.milestone_dir.glob('best_*.pt'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
            elif ctype == 'production':
                checkpoints = sorted(
                    self.production_dir.glob('bayesian_transformer_*.pt'),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )

            results[ctype] = checkpoints

        return results

    def cleanup_old_checkpoints(self, checkpoint_type: str, keep: int):
        """Manually cleanup old checkpoints.

        Args:
            checkpoint_type: Type of checkpoint to clean
            keep: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints(checkpoint_type)[checkpoint_type]

        for checkpoint in checkpoints[keep:]:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint.name}")

    def _cleanup_old_training_checkpoints(self):
        """Keep only last N training checkpoints."""
        checkpoints = sorted(
            self.training_dir.glob('checkpoint_*.pt'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove oldest if exceeds limit
        for checkpoint in checkpoints[self.max_training_checkpoints:]:
            checkpoint.unlink()

    def _update_symlink(self, link_path: Path, target_name: str):
        """Update symlink atomically (cross-platform compatible)."""
        # Remove existing symlink/file
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()

        try:
            # Try creating symlink (Unix/Linux)
            link_path.symlink_to(target_name)
        except OSError:
            # Fallback to copy on Windows if symlinks not supported
            target_path = link_path.parent / target_name
            if target_path.exists():
                shutil.copy(target_path, link_path)

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return None

    def _get_metadata(self) -> Dict[str, Any]:
        """Get environment metadata."""
        metadata = {
            'pytorch_version': torch.__version__,
            'python_version': sys.version,
            'hostname': socket.gethostname(),
            'timestamp': datetime.now().isoformat()
        }

        # Add CUDA info if available
        if torch.cuda.is_available():
            metadata['device'] = torch.cuda.get_device_name(0)
            metadata['cuda_version'] = torch.version.cuda
            metadata['cudnn_version'] = torch.backends.cudnn.version()
        else:
            metadata['device'] = 'CPU'

        return metadata

    def _save_model_card(self, version: str, model_card: Dict[str, Any],
                        metrics: Dict[str, float]):
        """Save model card as YAML."""
        model_cards_dir = self.metadata_dir / 'model_cards'
        model_cards_dir.mkdir(exist_ok=True)

        card_path = model_cards_dir / f'v{version}.yaml'

        # Combine model card with metrics
        full_card = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            **model_card
        }

        # Save as JSON (YAML requires extra dependency)
        card_json_path = model_cards_dir / f'v{version}.json'
        with open(card_json_path, 'w') as f:
            json.dump(full_card, f, indent=2)

        print(f"  Model card saved: {card_json_path.name}")


def resume_training(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Helper function to resume training from checkpoint.

    Args:
        checkpoint_path: Path to training checkpoint
        model: Model instance to restore
        optimizer: Optimizer instance to restore
        scheduler: Scheduler instance to restore
        device: Device to load to

    Returns:
        Dictionary with training state (epoch, step, metrics)

    Example:
        >>> model = BayesianTransformerModel(config)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> state = resume_training('checkpoints/training/latest.pt',
        ...                         model, optimizer)
        >>>
        >>> start_epoch = state['epoch'] + 1
        >>> print(f"Resuming from epoch {start_epoch}")
    """
    manager = CheckpointManager()
    state = manager.load_checkpoint(checkpoint_path, model, optimizer,
                                   scheduler, device)

    print(f"\nTraining resumed from checkpoint:")
    print(f"  Epoch: {state['epoch']}")
    print(f"  Step: {state['global_step']}")
    print(f"  Previous metrics: {state['metrics']}")

    return state
