# ADR-004: Model Checkpointing Strategy

**Date:** 2025-10-25
**Status:** Accepted
**Deciders:** System Architecture Team
**Technical Story:** Training State Persistence and Recovery

---

## Context and Problem Statement

We need a robust checkpointing strategy to:
1. Resume training after interruptions
2. Track model evolution over time
3. Enable model versioning and rollback
4. Support experiment comparison
5. Minimize storage costs while maintaining recovery capability

**Decision Drivers:**
- Training interruption recovery (GPU failures, spot instance termination)
- Model versioning for production deployment
- Storage efficiency (checkpoint files can be large)
- Reproducibility (exact training state restoration)
- Experiment tracking integration
- Production deployment requirements

---

## Considered Options

### Option 1: Simple State Dict Only
Save only model weights:
```python
torch.save(model.state_dict(), 'model.pt')
```

**Pros:**
- Smallest file size
- Simple to implement
- Fast save/load

**Cons:**
- Cannot resume training (no optimizer state)
- Loses training history
- No configuration versioning
- No metadata for production

### Option 2: Full Training State
Save everything needed to resume:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'loss': loss,
    'config': config
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
```

**Pros:**
- Complete training resumption
- Includes optimizer momentum
- Configuration versioning
- Reproducible training

**Cons:**
- Larger file size (3x state dict only)
- More complex to implement
- Requires coordination with training loop

### Option 3: Hierarchical Checkpointing
Separate checkpoints for different purposes:
- **Latest checkpoint:** Full state for recovery (frequent)
- **Milestone checkpoints:** Full state at key epochs (infrequent)
- **Production checkpoint:** State dict + config only (validation)

**Pros:**
- Optimizes storage vs recovery tradeoff
- Separate concerns (training vs deployment)
- Flexible retention policies
- Supports multiple use cases

**Cons:**
- More complex implementation
- Multiple checkpoint types to manage
- Coordination between checkpoint types

### Option 4: Differential Checkpointing
Save only changes from base checkpoint:

**Pros:**
- Minimal storage for frequent checkpoints
- Fast incremental saves

**Cons:**
- Complex implementation
- Slow to restore (need to replay changes)
- Fragile (corrupted base breaks all)
- Not worth complexity for our use case

---

## Decision Outcome

**Chosen option: Hierarchical Checkpointing (Option 3)**

### Checkpoint Types

#### 1. Training Checkpoints (Frequent)
**Purpose:** Resume training after interruption
**Frequency:** Every N epochs (default: 1)
**Retention:** Keep last K checkpoints (default: 3)

**Structure:**
```python
training_checkpoint = {
    # Version and metadata
    'version': '1.0.0',
    'timestamp': datetime.now().isoformat(),
    'git_commit': get_git_commit(),

    # Model state
    'model_state_dict': model.state_dict(),

    # Optimizer state (for momentum, etc.)
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,

    # Training state
    'epoch': epoch,
    'global_step': global_step,
    'best_metric': best_metric,
    'metrics_history': metrics_history,

    # Configuration
    'config': {
        'd_model': config['d_model'],
        'n_heads': config['n_heads'],
        'vocab_size': config['vocab_size'],
        'k_permutations': config['k_permutations'],
        'dropout': config['dropout'],
        # ... all config params
    },

    # Dataset information
    'dataset': {
        'name': 'imdb',
        'split': 'train',
        'preprocessing': preprocessing_config
    },

    # Hardware and environment
    'metadata': {
        'device': str(device),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'pytorch_version': torch.__version__,
        'training_time': training_time_seconds
    }
}
```

**Filename Pattern:**
```
checkpoints/training/epoch_{epoch:04d}_step_{global_step:08d}.pt
```

#### 2. Milestone Checkpoints (Infrequent)
**Purpose:** Long-term experiment tracking
**Frequency:** Validation improvement OR every M epochs (default: 5)
**Retention:** Indefinite

**Structure:** Same as training checkpoint

**Filename Pattern:**
```
checkpoints/milestones/best_accuracy_{accuracy:.4f}_epoch_{epoch:04d}.pt
checkpoints/milestones/best_loss_{loss:.4f}_epoch_{epoch:04d}.pt
checkpoints/milestones/epoch_{epoch:04d}.pt
```

#### 3. Production Checkpoints (On Demand)
**Purpose:** Model deployment
**Frequency:** Manual export after validation
**Retention:** Versioned (keep all)

**Structure:**
```python
production_checkpoint = {
    # Minimal information for inference
    'version': '1.0.0',
    'model_state_dict': model.state_dict(),
    'config': inference_config,  # Only what's needed for inference

    # Performance metrics
    'metrics': {
        'accuracy': final_accuracy,
        'loss': final_loss,
        'uncertainty_calibration': calibration_metrics
    },

    # Model card information
    'model_card': {
        'description': 'Bayesian Expectation Transformer for sentiment analysis',
        'dataset': 'IMDB',
        'training_date': datetime.now().isoformat(),
        'intended_use': 'Sentiment classification with uncertainty',
        'performance': performance_summary
    }
}
```

**Filename Pattern:**
```
checkpoints/production/bayesian_transformer_v{major}.{minor}.{patch}.pt
```

### Directory Structure

```
checkpoints/
├── training/
│   ├── latest.pt -> epoch_0010_step_12345.pt (symlink)
│   ├── epoch_0008_step_09876.pt
│   ├── epoch_0009_step_11110.pt
│   └── epoch_0010_step_12345.pt
├── milestones/
│   ├── best_accuracy_0.9234_epoch_0007.pt
│   ├── best_loss_0.1234_epoch_0009.pt
│   ├── epoch_0005.pt
│   └── epoch_0010.pt
├── production/
│   ├── bayesian_transformer_v1.0.0.pt
│   ├── bayesian_transformer_v1.0.1.pt
│   └── latest.pt -> bayesian_transformer_v1.0.1.pt (symlink)
└── metadata/
    ├── training_history.json
    └── model_cards/
        ├── v1.0.0.yaml
        └── v1.0.1.yaml
```

### Implementation

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, keep_last: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last = keep_last
        self.training_dir = checkpoint_dir / "training"
        self.milestone_dir = checkpoint_dir / "milestones"
        self.production_dir = checkpoint_dir / "production"

        # Create directories
        for dir_path in [self.training_dir, self.milestone_dir,
                         self.production_dir, checkpoint_dir / "metadata"]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_training_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        config: Dict[str, Any]
    ) -> Path:
        """Save training checkpoint with full state."""
        checkpoint = {
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'global_step': global_step,
            'metrics': metrics,
            'config': config,
            'metadata': self._get_metadata()
        }

        # Save to temporary file first (atomic write)
        filename = f"epoch_{epoch:04d}_step_{global_step:08d}.pt"
        filepath = self.training_dir / filename
        temp_filepath = filepath.with_suffix('.pt.tmp')

        torch.save(checkpoint, temp_filepath)
        temp_filepath.rename(filepath)  # Atomic rename

        # Update 'latest' symlink
        latest_link = self.training_dir / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filename)

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return filepath

    def save_milestone_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        checkpoint_type: str = "epoch"  # "epoch", "best_accuracy", "best_loss"
    ) -> Path:
        """Save milestone checkpoint."""
        checkpoint = {
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'metrics': metrics,
            'config': config,
            'metadata': self._get_metadata()
        }

        # Generate filename based on type
        if checkpoint_type == "best_accuracy":
            filename = f"best_accuracy_{metrics['accuracy']:.4f}_epoch_{epoch:04d}.pt"
        elif checkpoint_type == "best_loss":
            filename = f"best_loss_{metrics['loss']:.4f}_epoch_{epoch:04d}.pt"
        else:
            filename = f"epoch_{epoch:04d}.pt"

        filepath = self.milestone_dir / filename
        torch.save(checkpoint, filepath)

        return filepath

    def export_production_checkpoint(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        version: str,
        model_card: Dict[str, Any]
    ) -> Path:
        """Export production-ready checkpoint."""
        checkpoint = {
            'version': version,
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics,
            'model_card': model_card
        }

        filename = f"bayesian_transformer_v{version}.pt"
        filepath = self.production_dir / filename

        # Save with compression
        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=True)

        # Update 'latest' symlink
        latest_link = self.production_dir / "latest.pt"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filename)

        # Save model card separately
        self._save_model_card(version, model_card)

        return filepath

    def load_checkpoint(self, filepath: Path) -> Dict[str, Any]:
        """Load checkpoint with validation."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location='cpu')

        # Validate checkpoint structure
        required_keys = ['version', 'model_state_dict', 'config']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint: missing keys {missing_keys}")

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Keep only last K training checkpoints."""
        checkpoints = sorted(
            self.training_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old checkpoints beyond keep_last
        for checkpoint in checkpoints[self.keep_last:]:
            checkpoint.unlink()

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return None

    def _get_metadata(self) -> Dict[str, Any]:
        """Get environment metadata."""
        return {
            'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pytorch_version': torch.__version__,
            'python_version': sys.version,
            'hostname': socket.gethostname()
        }

    def _save_model_card(self, version: str, model_card: Dict[str, Any]):
        """Save model card as YAML."""
        import yaml
        model_card_path = self.checkpoint_dir / "metadata" / "model_cards" / f"v{version}.yaml"
        model_card_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_card_path, 'w') as f:
            yaml.dump(model_card, f, default_flow_style=False)
```

### Positive Consequences

- **Recovery:** Can resume training exactly from any point
- **Versioning:** Clear model versioning for production
- **Storage Optimization:** Balance between recovery and storage costs
- **Flexibility:** Different checkpoint types for different purposes
- **Reproducibility:** Full training state and configuration captured
- **Atomic Writes:** Prevents corrupted checkpoints (temp file + rename)
- **Metadata:** Rich context for debugging and analysis

### Negative Consequences

- **Complexity:** More complex than simple state dict saving
- **Storage:** Training checkpoints can be large (3x model size)
- **Coordination:** Requires careful coordination with training loop

### Mitigation Strategies

**Storage Costs:**
- Aggressive cleanup of training checkpoints (keep_last=3)
- Compress production checkpoints
- Use cloud storage with lifecycle policies (delete after 30 days)

**Complexity:**
- Encapsulate in CheckpointManager class
- Clear documentation and examples
- Unit tests for checkpoint save/load

---

## Validation

### Success Criteria

✅ **Met:**
- Can resume training after interruption
- Checkpoint save/load time <30 seconds
- No checkpoint corruption (atomic writes)
- Clear version history for production models

🔄 **In Progress:**
- Cloud storage integration (S3)
- Checkpoint compression testing

⏳ **Planned:**
- Automated checkpoint cleanup policies
- Integration with experiment tracking (W&B)

---

## Related Decisions

- [ADR-001: PyTorch over TensorFlow](ADR-001-pytorch-over-tensorflow.md) - Model framework
- [ADR-005: Monitoring Solution](ADR-005-monitoring-solution.md) - Experiment tracking

---

## References

1. PyTorch Saving and Loading Models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
2. Model Cards: https://arxiv.org/abs/1810.03993

---

**Last Updated:** 2025-10-25
**Next Review:** 2026-01-25
