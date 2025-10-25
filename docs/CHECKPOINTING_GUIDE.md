# Model Checkpointing System Guide

## Overview

The Bayesian Expectation Transformer project implements a robust 3-tier checkpointing system designed for production-ready model training, versioning, and deployment.

## Architecture

### 3-Tier Checkpointing Strategy

```
checkpoints/
├── training/          # Frequent saves, rolling retention
│   ├── checkpoint_epoch_0001_step_00000100.pt
│   ├── checkpoint_epoch_0001_step_00000200.pt
│   ├── checkpoint_epoch_0001_step_00000300.pt
│   └── latest.pt -> checkpoint_epoch_0001_step_00000300.pt
│
├── milestone/         # On improvement, indefinite retention
│   ├── best_accuracy_0.8500_epoch_0007.pt
│   ├── best_accuracy_0.8500_epoch_0007.json
│   ├── best_loss_0.1234_epoch_0009.pt
│   └── best_accuracy.pt -> best_accuracy_0.8500_epoch_0007.pt
│
├── production/        # Versioned releases
│   ├── bayesian_transformer_v1.0.0.pt
│   ├── bayesian_transformer_v1.0.0.config.json
│   ├── bayesian_transformer_v1.1.0.pt
│   └── latest.pt -> bayesian_transformer_v1.1.0.pt
│
└── metadata/
    └── model_cards/
        ├── v1.0.0.json
        └── v1.1.0.json
```

## Quick Start

### Basic Usage

```python
from src.bayesian_transformer import CheckpointManager

# Initialize checkpoint manager
manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    max_training_checkpoints=3,  # Keep last 3
    save_optimizer=True
)

# During training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # ... training code ...

        # Save every 100 steps
        if step % 100 == 0:
            manager.save_training_checkpoint(
                model, optimizer,
                epoch=epoch,
                step=step,
                metrics={'loss': loss.item(), 'accuracy': acc}
            )

    # After validation
    val_acc = validate(model, val_loader)

    if val_acc > best_acc:
        manager.save_milestone_checkpoint(
            model,
            metric_value=val_acc,
            metric_name='accuracy',
            epoch=epoch,
            config=model_config
        )
        best_acc = val_acc

# Save final production version
manager.save_production_checkpoint(
    model,
    version='1.0.0',
    config=model_config,
    metrics={'accuracy': best_acc},
    model_card={
        'description': 'Production model for sentiment analysis',
        'dataset': 'IMDB',
        'performance': {'accuracy': best_acc}
    }
)
```

### Resume Training

```python
from src.bayesian_transformer import resume_training

# Load latest checkpoint
state = resume_training(
    'checkpoints/training/latest.pt',
    model,
    optimizer,
    scheduler
)

# Continue training from saved state
start_epoch = state['epoch'] + 1
print(f"Resuming from epoch {start_epoch}")
print(f"Previous metrics: {state['metrics']}")

for epoch in range(start_epoch, num_epochs):
    # Continue training...
    pass
```

## Checkpoint Types

### 1. Training Checkpoints

**Purpose:** Enable exact training resumption after interruptions

**Features:**
- Full training state (model, optimizer, scheduler)
- Saved periodically (configurable interval)
- Rolling retention (keeps last N)
- Atomic writes (no corruption)

**Contents:**
```python
{
    'version': '1.0.0',
    'checkpoint_type': 'training',
    'timestamp': '2025-10-25T13:30:00',
    'git_commit': 'abc123...',

    # Full state
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},

    # Training progress
    'epoch': 5,
    'global_step': 1000,
    'metrics': {'loss': 0.25, 'accuracy': 0.85},

    # Configuration
    'config': {'d_model': 512, ...},

    # Environment
    'metadata': {
        'device': 'NVIDIA RTX 4090',
        'pytorch_version': '2.6.0',
        ...
    }
}
```

### 2. Milestone Checkpoints

**Purpose:** Track model evolution and improvements

**Features:**
- Saved when validation metric improves
- Indefinite retention
- Includes metadata and notes
- Separate JSON file for easy inspection

**Usage:**
```python
# Saves only if accuracy improved
path = manager.save_milestone_checkpoint(
    model,
    metric_value=0.92,
    metric_name='accuracy',
    epoch=7,
    config=model_config,
    metadata={
        'notes': 'Best model so far',
        'training_hours': 12.5,
        'dataset_version': '2.0'
    }
)

if path:
    print(f"New best model saved: {path}")
```

### 3. Production Checkpoints

**Purpose:** Deployment-ready model versions

**Features:**
- Semantic versioning (e.g., '1.0.0')
- Minimal size (no optimizer state)
- Comprehensive model card
- Config saved as JSON

**Usage:**
```python
manager.save_production_checkpoint(
    model,
    version='1.0.0',
    config={
        'd_model': 512,
        'n_heads': 8,
        'vocab_size': 50000
    },
    metrics={
        'accuracy': 0.92,
        'f1_score': 0.91,
        'inference_latency_ms': 45
    },
    model_card={
        'description': 'Bayesian Transformer for sentiment analysis',
        'dataset': 'IMDB (50K reviews)',
        'intended_use': 'Binary sentiment classification',
        'training_epochs': 10,
        'limitations': 'English text only',
        'ethical_considerations': 'May reflect dataset biases'
    }
)
```

## Advanced Features

### Checkpoint Management

```python
# List all checkpoints
all_checkpoints = manager.list_checkpoints()
print(f"Training: {len(all_checkpoints['training'])}")
print(f"Milestone: {len(all_checkpoints['milestone'])}")
print(f"Production: {len(all_checkpoints['production'])}")

# Get latest checkpoint
latest_training = manager.get_latest_checkpoint('training')
latest_production = manager.get_latest_checkpoint('production')

# Manual cleanup
manager.cleanup_old_checkpoints('training', keep=2)
```

### Custom Configuration

```python
manager = CheckpointManager(
    checkpoint_dir='custom/path',
    max_training_checkpoints=5,  # Keep last 5
    save_optimizer=True,         # Save optimizer state
    save_scheduler=True,         # Save scheduler state
    compress=True                # Enable compression
)
```

### Loading Checkpoints

```python
# Load with full state restoration
metadata = manager.load_checkpoint(
    'checkpoints/milestone/best_accuracy_0.92.pt',
    model,
    optimizer,
    scheduler,
    device='cuda'
)

print(f"Loaded checkpoint from epoch {metadata['epoch']}")
print(f"Metrics: {metadata['metrics']}")
```

## Best Practices

### 1. Training Checkpoints

```python
# Save frequently enough to recover from failures
# But not so frequently that it slows training
save_interval = 100  # steps

# Adjust based on training speed
if steps_per_epoch > 1000:
    save_interval = 200  # Less frequent for long epochs
```

### 2. Milestone Checkpoints

```python
# Track multiple metrics
metrics_to_track = ['accuracy', 'loss', 'f1_score']

for metric_name in metrics_to_track:
    metric_value = evaluate(model, val_loader, metric_name)
    manager.save_milestone_checkpoint(
        model, metric_value, metric_name, epoch, config
    )
```

### 3. Production Checkpoints

```python
# Use semantic versioning
version = '1.0.0'  # Major.Minor.Patch
# - Major: Breaking changes
# - Minor: New features, backward compatible
# - Patch: Bug fixes

# Include comprehensive model card
model_card = {
    'description': 'Clear description',
    'dataset': 'Name and size',
    'intended_use': 'Primary use case',
    'training_epochs': 10,
    'training_time_hours': 24,
    'performance': {
        'accuracy': 0.92,
        'precision': 0.91,
        'recall': 0.90
    },
    'limitations': [
        'English text only',
        'Max 512 tokens'
    ],
    'ethical_considerations': [
        'May reflect training data biases',
        'Not suitable for medical decisions'
    ]
}
```

## Testing

The checkpointing system includes comprehensive tests:

```bash
# Run all checkpointing tests
pytest tests/test_checkpointing.py -v

# Run specific test category
pytest tests/test_checkpointing.py::TestTrainingCheckpoints -v
pytest tests/test_checkpointing.py::TestMilestoneCheckpoints -v
pytest tests/test_checkpointing.py::TestProductionCheckpoints -v
```

## Integration with Training Scripts

See `examples/checkpointing_demo.py` for a complete integration example.

## Deployment Workflow

### Development → Staging → Production

```python
# 1. Development: Train with checkpointing
manager = CheckpointManager('checkpoints/dev')
# ... training code ...

# 2. Staging: Export best milestone
best_milestone = manager.get_latest_checkpoint('milestone')
staging_model = load_model(best_milestone)
staging_metrics = validate_extensive(staging_model)

# 3. Production: Create versioned release
if staging_metrics['accuracy'] > 0.90:
    prod_manager = CheckpointManager('checkpoints/prod')
    prod_manager.save_production_checkpoint(
        staging_model,
        version='1.1.0',
        config=model_config,
        metrics=staging_metrics,
        model_card=create_model_card(staging_metrics)
    )
```

## Recovery Scenarios

### Scenario 1: Training Interrupted

```python
# Check for existing checkpoint
latest = manager.get_latest_checkpoint('training')

if latest:
    print(f"Resuming from {latest}")
    state = resume_training(latest, model, optimizer)
    start_epoch = state['epoch'] + 1
else:
    print("Starting fresh training")
    start_epoch = 0

# Continue training
for epoch in range(start_epoch, num_epochs):
    train_one_epoch(...)
```

### Scenario 2: Model Degradation

```python
# Compare current model with previous milestone
current_acc = validate(current_model, val_loader)
milestone_path = 'checkpoints/milestone/best_accuracy_0.92.pt'
milestone_model = load_model(milestone_path)
milestone_acc = validate(milestone_model, val_loader)

if current_acc < milestone_acc - 0.05:  # 5% degradation
    print("WARNING: Model degraded, rolling back")
    model = milestone_model
```

### Scenario 3: Production Rollback

```python
# Issue with v1.1.0, rollback to v1.0.0
manager.load_checkpoint(
    'checkpoints/production/bayesian_transformer_v1.0.0.pt',
    model
)

# Deploy previous version
deploy_model(model, version='1.0.0')
```

## Performance Considerations

### Checkpoint Size

```python
# Training checkpoint: ~3x model size
# - Model weights: 1x
# - Optimizer state: 2x (Adam with momentum)

# Milestone checkpoint: ~3x model size
# - Full state for resume capability

# Production checkpoint: ~1x model size
# - Model weights only
# - Compressed for distribution
```

### Save/Load Speed

```python
# Typical save times (V100 GPU, 100M params):
# - Training checkpoint: 5-10 seconds
# - Milestone checkpoint: 5-10 seconds
# - Production checkpoint: 2-5 seconds (no optimizer)

# Typical load times:
# - Any checkpoint: 2-5 seconds
```

## References

- [ADR-004: Model Checkpointing Strategy](../adrs/ADR-004-model-checkpointing-strategy.md)
- [System Architecture](SYSTEM_ARCHITECTURE.md) - Section 3.3
- [Examples](../examples/checkpointing_demo.py)
- [Tests](../tests/test_checkpointing.py)

## Support

For issues or questions:
1. Check test examples: `tests/test_checkpointing.py`
2. Review demo: `examples/checkpointing_demo.py`
3. Read ADR-004 for design rationale
4. Create an issue on GitHub
