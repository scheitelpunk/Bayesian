# Transformer Benchmark Report
**Date**: 2025-10-27 02:31:13
**Device**: cpu
**Dataset**: IMDB (max 20000 samples)

## Summary Comparison

| Metric | Standard Transformer | Bayesian Transformer | Winner |
|--------|----------------------|----------------------|--------|
| **Accuracy** | 0.7956 | 0.9032 | Bayesian |
| **Training Speed** | 120.27s/epoch | 1062.64s/epoch | Standard |
| **Inference Speed** | 0.47ms | 10.20ms | Standard |
| **Memory Usage** | 50.39MB | 273.34MB | Standard |
| **Parameters** | 1,544,066 | 4,430,185 | Standard |
| **Convergence (80% acc)** | Epoch 2 | Epoch 2 | Bayesian |
| **Uncertainty Quantification** | [X] Not available | [OK] Available (mean: 0.6082) | Bayesian |

## Detailed Analysis

### Training Performance

- **Standard Transformer**: 120.27s per epoch
- **Bayesian Transformer**: 1062.64s per epoch
- **Overhead**: 783.5% slower

### Inference Performance

- **Standard Transformer**: 0.47ms per sample
- **Bayesian Transformer**: 10.20ms per sample
- **Overhead**: 2067.9% slower

### Uncertainty Quantification (Bayesian Only)

- **Mean Uncertainty**: 0.6082
- **Uncertainty-Error Correlation**: 0.0000
- [WARN] **Weak correlation**: Uncertainty may need calibration

### Key Findings

[OK] **Bayesian Transformer achieves higher accuracy**

[WARN] **Significant overhead** (>50% slower than standard)

### Recommendations

**Use Standard Transformer when:**
- Speed is critical and uncertainty is not needed
- Memory is constrained
- Simple classification without confidence scores

