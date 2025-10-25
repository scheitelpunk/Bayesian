# TensorBoard Integration Guide

## Overview

The Bayesian Transformer project includes comprehensive TensorBoard logging for monitoring training progress, debugging issues, and analyzing model behavior.

## Quick Start

### 1. Installation

TensorBoard is already included in `requirements.txt`:

```bash
pip install tensorboard>=2.10.0
```

### 2. Basic Usage

```python
from src.bayesian_transformer import BayesianTransformerLogger

# Create logger with context manager
with BayesianTransformerLogger(
    log_dir='runs',
    experiment_name='my_experiment'
) as logger:

    for epoch in range(num_epochs):
        # Training loop
        for batch in dataloader:
            # ... forward/backward pass ...

            # Log metrics
            logger.log_metrics({
                'loss': loss.item(),
                'accuracy': accuracy,
            }, prefix='train')

            # Log learning rate
            logger.log_learning_rate(optimizer)

            # Log gradients (every 100 steps by default)
            logger.log_gradients(model)

            # Log uncertainty metrics
            logger.log_uncertainty_metrics(epistemic_uncertainty)

            # Log histograms (every 500 steps by default)
            logger.log_model_histograms(model)

            logger.increment_step()
```

### 3. View Logs

**Windows:**
```bash
scripts\view_tensorboard.bat
```

**Linux/Mac:**
```bash
bash scripts/view_tensorboard.sh
```

**Manual:**
```bash
tensorboard --logdir=runs --port=6006
```

Then navigate to: http://localhost:6006

## Features

### 1. Training Metrics

Log loss, accuracy, and custom metrics:

```python
logger.log_metrics({
    'loss': 0.345,
    'accuracy': 0.912,
    'f1_score': 0.895,
}, prefix='train', step=100)
```

Metrics are organized by prefix:
- `train/*` - Training metrics
- `val/*` - Validation metrics
- `test/*` - Test metrics

### 2. Gradient Monitoring

Detect vanishing/exploding gradients:

```python
logger.log_gradients(model, step=100)
```

Logs:
- Per-parameter gradient norms
- Layer-wise average gradient norms
- Total gradient norm

**Configuration:**
```python
logger = BayesianTransformerLogger(
    log_gradients_every=100  # Log every 100 steps
)
```

### 3. Learning Rate Tracking

Monitor learning rate schedules:

```python
logger.log_learning_rate(optimizer, step=100)
```

Useful for debugging convergence issues and validating scheduler behavior.

### 4. Attention Statistics

Monitor attention patterns to detect attention collapse:

```python
logger.log_attention_stats(
    attention_weights,  # [batch, heads, seq_len, seq_len]
    attention_entropy,  # Computed entropy
    step=100
)
```

Logs:
- Average attention entropy
- Attention weight mean/std
- Per-layer statistics

### 5. Uncertainty Quantification

Track epistemic and aleatoric uncertainty:

```python
logger.log_uncertainty_metrics(
    epistemic_uncertainty,
    aleatoric_uncertainty,  # Optional
    step=100
)
```

Logs:
- Mean uncertainty values
- Uncertainty distributions (histograms)

### 6. Parameter Histograms

Visualize parameter and gradient distributions:

```python
logger.log_model_histograms(model, step=500)
```

**Configuration:**
```python
logger = BayesianTransformerLogger(
    log_histograms_every=500  # Log every 500 steps
)
```

### 7. Text Logging

Log predictions, errors, or any text:

```python
logger.log_text(
    'predictions',
    'Sample: This movie was great! -> Positive (0.95 confidence)',
    step=100
)
```

## Configuration Options

```python
logger = BayesianTransformerLogger(
    log_dir='runs',                    # Base log directory
    experiment_name='experiment_1',    # Subdirectory name
    log_gradients_every=100,           # Gradient logging frequency
    log_histograms_every=500           # Histogram logging frequency
)
```

**Default experiment name:** `YYYYMMDD_HHMMSS` (timestamp)

## Integration with Training

### Full Training Loop Example

```python
from src.bayesian_transformer import BayesianTransformerLogger, CheckpointManager

# Initialize
checkpoint_mgr = CheckpointManager('checkpoints', max_keep=3)

with BayesianTransformerLogger(
    log_dir='runs',
    experiment_name='bayesian_transformer_imdb'
) as logger:

    print(f"Logs: {logger.log_dir}")
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            outputs = model(batch['input_ids'], return_uncertainty=True)
            loss = criterion(outputs['logits'], batch['labels'])

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            logger.log_metrics({'loss': loss.item()}, prefix='train')
            logger.log_learning_rate(optimizer)
            logger.log_gradients(model)

            if 'epistemic_uncertainty' in outputs:
                logger.log_uncertainty_metrics(
                    outputs['epistemic_uncertainty']
                )

            logger.log_model_histograms(model)
            logger.increment_step()

        # Validation
        val_acc = evaluate(model, val_loader)
        logger.log_metrics({'accuracy': val_acc}, step=epoch, prefix='val')

        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_mgr.save_checkpoint(
                model, optimizer, None, epoch, logger.global_step,
                {'val_accuracy': val_acc},
                is_best=True
            )

    print(f"\nView logs: tensorboard --logdir={logger.log_dir}")
```

## TensorBoard Dashboard Sections

### 1. Scalars
- Training loss/accuracy curves
- Validation metrics
- Learning rate schedules
- Gradient norms
- Uncertainty metrics

### 2. Histograms
- Parameter distributions
- Gradient distributions
- Activation distributions

### 3. Text
- Sample predictions
- Error analysis
- Model outputs

## Best Practices

### 1. Experiment Organization

Use descriptive experiment names:

```python
experiment_name = f'imdb_bs32_lr1e-3_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
logger = BayesianTransformerLogger(experiment_name=experiment_name)
```

### 2. Logging Frequency

Balance between detail and performance:

- **Metrics**: Every batch (cheap)
- **Gradients**: Every 100-1000 steps (moderate cost)
- **Histograms**: Every 500-1000 steps (expensive)

```python
logger = BayesianTransformerLogger(
    log_gradients_every=100,
    log_histograms_every=500
)
```

### 3. Monitor Gradient Health

Watch for:
- **Vanishing gradients**: Norms approaching 0
- **Exploding gradients**: Norms > 10
- **Gradient collapse**: All layers same norm

### 4. Attention Collapse Detection

Monitor attention entropy:
- **Healthy**: Entropy between 0.5-2.0
- **Collapsed**: Entropy < 0.1 (attention focuses on single token)

### 5. Uncertainty Calibration

Track uncertainty trends:
- **Well-calibrated**: Uncertainty correlates with errors
- **Under-confident**: High uncertainty on correct predictions
- **Over-confident**: Low uncertainty on errors

## Troubleshooting

### TensorBoard Not Starting

```bash
# Check if port 6006 is in use
netstat -ano | findstr :6006

# Use different port
tensorboard --logdir=runs --port=6007
```

### No Data Showing

1. Check log directory exists: `runs/experiment_name/`
2. Verify event files: `events.out.tfevents.*`
3. Refresh browser (Ctrl+F5)
4. Clear browser cache

### Large Log Files

```python
# Reduce logging frequency
logger = BayesianTransformerLogger(
    log_gradients_every=1000,  # Instead of 100
    log_histograms_every=5000  # Instead of 500
)
```

### Memory Issues

TensorBoard loads all events into memory:

```bash
# Limit samples
tensorboard --logdir=runs --samples_per_plugin=1000
```

## Advanced Features

### Multiple Experiments Comparison

```bash
# Compare multiple runs
tensorboard --logdir=runs
```

TensorBoard automatically groups experiments from the same `log_dir`.

### HParams Dashboard

Track hyperparameter tuning:

```python
# Log hyperparameters with final metrics
logger.writer.add_hparams(
    {
        'lr': 1e-3,
        'batch_size': 32,
        'num_layers': 6,
    },
    {
        'final_accuracy': 0.93,
        'final_loss': 0.21,
    }
)
```

### Remote Access

```bash
# Allow external access
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```

Access from: `http://<server-ip>:6006`

## Integration with Research

Based on research findings in `docs/research/ml-deployment-research-2025.md`:

1. **Gradient Monitoring** (Section 3):
   - Detect vanishing/exploding gradients early
   - Log every 100-1000 steps

2. **Attention Entropy** (Section 3):
   - Monitor for attention collapse
   - Alert if entropy < 0.1

3. **Learning Rate Schedules** (Section 3):
   - Validate warmup/decay phases
   - Debug convergence issues

4. **Model Calibration** (Section 3):
   - Track uncertainty vs. accuracy
   - Identify over/under-confident predictions

## References

- Research: `docs/research/ml-deployment-research-2025.md` (Section 3)
- Code: `src/bayesian_transformer/monitoring.py`
- Tests: `tests/test_tensorboard_integration.py`
- Example: `examples/real_data_demo.py`

## See Also

- [Checkpointing Guide](./checkpointing-guide.md)
- [Training Guide](./training-guide.md)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
