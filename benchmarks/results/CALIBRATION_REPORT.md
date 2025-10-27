# Uncertainty Calibration Report

## Summary

Tested three calibration methods on validation set:

| Method | ECE (Before) | ECE (After) | Delta ECE | Corr (Before) | Corr (After) | Delta Corr |
|--------|--------------|-------------|-----------|---------------|--------------|------------|
| Temperature | 0.6582 | 0.5391 | 0.1191 | 0.0731 | 0.0731 | 0.0000 |
| Platt | 0.6582 | 0.0099 | 0.6483 | 0.0731 | 0.0823 | 0.0093 |
| Isotonic | 0.6582 | 0.0228 | 0.6354 | 0.0731 | 0.0930 | 0.0199 |

**Best Method**: Platt

## Best Method Results

- **ECE**: 0.6582 -> 0.0099 (0.6483 improvement)
- **MCE**: 0.7335 -> 0.0164 (0.7170 improvement)
- **Correlation**: 0.0731 -> 0.0823 (0.0093 improvement)

[X] **Needs more work**: Correlation still < 0.3

## Recommendations

1. Use **platt scaling** for production
2. Apply calibration as post-processing step after inference
3. Re-calibrate when retraining model
4. Monitor ECE/MCE metrics in production

