# Benchmarking Guide

## Overview

This guide explains how to run comprehensive benchmarks comparing the Bayesian Transformer against standard PyTorch Transformers.

## Quick Start

```bash
# Run complete benchmark suite
python benchmarks/benchmark_transformer_comparison.py
```

## What Gets Benchmarked

### 1. Performance Metrics

- **Training Speed**: Time per epoch (seconds)
- **Inference Speed**: Time per sample (milliseconds)
- **Memory Usage**: RAM consumption during training (MB)
- **Model Size**: Total trainable parameters

### 2. Quality Metrics

- **Accuracy**: Final test set accuracy
- **Convergence Speed**: Epoch when accuracy reaches 80%
- **Loss Trajectory**: Training loss over epochs

### 3. Uncertainty Metrics (Bayesian Only)

- **Uncertainty Quantification**: Epistemic and aleatoric uncertainty
- **Calibration**: Correlation between uncertainty and prediction errors
- **Coverage**: Percentage of samples identified as uncertain

## Benchmark Results

After running, you'll find:

```
benchmarks/results/
├── benchmark_results.json       # Raw data (JSON)
├── BENCHMARK_REPORT.md         # Markdown report
├── benchmark_comparison.png    # Training curves & metrics
└── speedup_analysis.png        # Speed comparison
```

## Interpreting Results

### Accuracy Comparison

```markdown
| Metric | Standard | Bayesian | Winner |
|--------|----------|----------|--------|
| Accuracy | 0.8750 | 0.8825 | Bayesian |
```

**Higher is better**. If Bayesian accuracy is within 1% of Standard, it's a "Tie".

### Speed Comparison

```markdown
| Training Speed | 12.5s/epoch | 18.2s/epoch | Standard |
| Inference Speed | 2.1ms | 3.4ms | Standard |
```

**Lower is better**. Bayesian is typically 30-50% slower due to uncertainty computation.

### Uncertainty Analysis

```markdown
| Mean Uncertainty | - | 0.1234 | Bayesian |
| Uncertainty-Error Correlation | - | 0.67 | Bayesian |
```

**Correlation > 0.3 = Good**: Model assigns higher uncertainty to incorrect predictions.

## Configuration

### Dataset Size

```python
benchmark = TransformerBenchmark(
    device='cpu',
    max_samples=1000  # Increase for more robust results
)
```

- **1,000 samples**: Quick test (~5 minutes)
- **5,000 samples**: Standard benchmark (~20 minutes)
- **20,000 samples**: Comprehensive (~60 minutes)

### Model Configuration

```python
config = {
    'd_model': 256,        # Embedding dimension
    'n_heads': 4,          # Attention heads
    'vocab_size': 10000,   # Vocabulary size
    'dropout': 0.3,        # Dropout rate
    'k_permutations': 10,  # Bayesian permutations
    'epsilon': 0.05        # Bayesian epsilon
}
```

**For faster benchmarks**: Reduce `d_model` and `k_permutations`
**For more accurate benchmarks**: Increase to production config (512 dim, 20 permutations)

### Training Parameters

```python
benchmark.run_benchmarks(n_epochs=5)  # Number of epochs
```

- **5 epochs**: Quick comparison
- **10 epochs**: Standard
- **20 epochs**: Thorough

## Custom Benchmarks

### Add Custom Model

```python
class MyCustomTransformer(nn.Module):
    def __init__(self, config):
        # Your implementation
        pass

    def forward(self, x, return_uncertainty=False):
        # Must return logits or dict with 'logits' key
        pass

# Benchmark it
benchmark.results['My Model'] = benchmark.train_and_evaluate(
    MyCustomTransformer(config),
    'My Model',
    train_data,
    test_data
)
```

### Custom Metrics

```python
# Add custom metric to BenchmarkResult dataclass
@dataclass
class BenchmarkResult:
    # ... existing fields ...
    my_custom_metric: float = 0.0

# Compute in train_and_evaluate()
result.my_custom_metric = compute_my_metric(model, test_data)
```

## Visualizations

### 1. Training Curves

**benchmark_comparison.png** shows:
- Top-left: Loss over epochs
- Top-right: Accuracy over epochs
- Bottom-left: Performance metrics bar chart
- Bottom-right: Final accuracy comparison

### 2. Speedup Analysis

**speedup_analysis.png** shows:
- Horizontal bar chart
- >1x = Bayesian faster
- <1x = Standard faster

## Performance Tips

### For Faster Benchmarks:

1. **Reduce dataset size**: `max_samples=500`
2. **Smaller model**: `d_model=128, n_heads=2`
3. **Fewer epochs**: `n_epochs=3`
4. **Smaller batch size**: `batch_size=4`

### For GPU Benchmarks:

```python
# Automatic GPU detection
benchmark = TransformerBenchmark(device='cuda')
```

Ensure PyTorch CUDA is installed:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Expected Results

### Typical Performance (1K samples, CPU):

| Metric | Standard | Bayesian |
|--------|----------|----------|
| **Accuracy** | 85-88% | 86-89% |
| **Training** | 10-15s/epoch | 15-22s/epoch |
| **Inference** | 1.5-2.5ms | 2.5-4ms |
| **Memory** | 150-200MB | 200-280MB |
| **Parameters** | 1.2M | 1.4M |

### Uncertainty Performance:

- **Mean Uncertainty**: 0.10-0.20 (well-calibrated)
- **Correlation**: 0.4-0.7 (strong)
- **High-Uncertainty Samples**: 10-20% (realistic)

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
benchmark.train_and_evaluate(..., batch_size=4)

# Or reduce model size
config['d_model'] = 128
config['k_permutations'] = 5
```

### Slow Training

```python
# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Or reduce samples
benchmark = TransformerBenchmark(max_samples=500)
```

### Low Accuracy

- Increase `n_epochs` (try 10)
- Increase `max_samples` (try 5000)
- Adjust learning rate
- Check data quality

## Advanced Analysis

### Statistical Significance

```python
# Run multiple trials
for trial in range(5):
    benchmark.run_benchmarks()
    # Average results across trials
```

### Cross-Validation

```python
# Modify load_imdb_data() to support k-fold
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    # Run benchmark on each fold
```

### Hyperparameter Sweep

```python
for dropout in [0.1, 0.2, 0.3]:
    for k_perm in [5, 10, 20]:
        config['dropout'] = dropout
        config['k_permutations'] = k_perm
        # Run benchmark
```

## Output Files

### benchmark_results.json

```json
{
  "Standard Transformer": {
    "model_name": "Standard Transformer",
    "training_time_per_epoch": 12.5,
    "inference_time_per_sample": 2.1,
    "final_accuracy": 0.875,
    ...
  },
  "Bayesian Transformer": {
    ...
  }
}
```

### BENCHMARK_REPORT.md

Formatted markdown with:
- Summary table
- Detailed analysis
- Key findings
- Recommendations

## CI/CD Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmark
on: [push]
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmarks
        run: python benchmarks/benchmark_transformer_comparison.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/results/
```

## Best Practices

1. **Consistent Environment**: Same device, PyTorch version across runs
2. **Multiple Trials**: Run 3-5 times, report mean ± std
3. **Warm-up**: Discard first epoch (initialization overhead)
4. **Fair Comparison**: Same hyperparameters, data, hardware
5. **Document Config**: Save config alongside results

## References

- [MLPerf Benchmarking](https://mlperf.org/)
- [PyTorch Benchmark Utils](https://pytorch.org/docs/stable/benchmark_utils.html)
- [Uncertainty Calibration](https://arxiv.org/abs/1706.04599)

---

**Next Steps**: See `NEXT_STEPS.md` for using benchmark results in publications and presentations.
