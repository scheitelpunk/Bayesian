# Production Integration Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-26
**Status**: Production Ready ✅

This guide shows you how to integrate the Bayesian Expectation Transformer into your production systems.

---

## Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/scheitelpunk/Bayesian-Expectation-Transformer.git
cd Bayesian-Expectation-Transformer

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.bayesian_transformer import BayesianExpectationTransformerLayer; print('OK')"
```

### 2. Basic Usage

```python
import torch
from src.bayesian_transformer import BayesianExpectationTransformerLayer

# Configuration
config = {
    'd_model': 128,
    'n_heads': 4,
    'vocab_size': 30522,  # BERT vocab size
    'k_permutations': 5,
    'dropout': 0.2
}

# Create layer
layer = BayesianExpectationTransformerLayer(config)

# Input (batch_size, seq_length, d_model)
x = torch.randn(8, 64, 128)

# Forward pass with uncertainty
output = layer(x, return_uncertainty=True)

print(f"Output shape: {output['hidden_states'].shape}")
print(f"Uncertainty available: {output.get('epistemic_uncertainty') is not None}")
```

### 3. Apply Calibration (Recommended)

```python
from src.bayesian_transformer.uncertainty_calibration import PlattScaling
import numpy as np

# After training, calibrate on validation set
calibrator = PlattScaling()

# Collect validation uncertainties and errors
val_uncertainties = []  # Your validation uncertainties
val_errors = []  # Binary: 0=correct, 1=error

calibrator.fit(
    np.array(val_uncertainties),
    np.array(val_errors)
)

# At inference time
raw_uncertainty = output.get('epistemic_uncertainty', torch.zeros(batch_size))
calibrated = calibrator.transform(raw_uncertainty.cpu().numpy())
```

---

## Integration Scenarios

### Scenario 1: Replace Existing Transformer Layer

**Use Case**: You have an existing transformer model and want to add Bayesian capabilities.

```python
import torch.nn as nn
from transformers import BertModel
from src.bayesian_transformer import BayesianExpectationTransformerLayer

class BayesianBERT(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super().__init__()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(pretrained_model)

        # Replace first encoder layer with Bayesian version
        config = {
            'd_model': self.bert.config.hidden_size,
            'n_heads': self.bert.config.num_attention_heads,
            'vocab_size': self.bert.config.vocab_size,
            'k_permutations': 5,
            'dropout': 0.1
        }

        self.bayesian_layer = BayesianExpectationTransformerLayer(config)

        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, return_uncertainty=False):
        # Get BERT embeddings
        embeddings = self.bert.embeddings(input_ids)

        # Pass through Bayesian layer
        bayesian_out = self.bayesian_layer(
            embeddings,
            mask=attention_mask,
            return_uncertainty=return_uncertainty
        )

        # Continue with remaining BERT layers
        hidden = bayesian_out['hidden_states']
        for layer in self.bert.encoder.layer[1:]:
            hidden = layer(hidden, attention_mask=attention_mask)[0]

        # Pooler and classification
        pooled = self.bert.pooler(hidden)
        logits = self.classifier(pooled)

        result = {'logits': logits}
        if return_uncertainty and 'epistemic_uncertainty' in bayesian_out:
            result['uncertainty'] = bayesian_out['epistemic_uncertainty']

        return result
```

### Scenario 2: Build Custom Model from Scratch

**Use Case**: You want full control and are building a new model.

```python
import torch.nn as nn
from src.bayesian_transformer import BayesianExpectationTransformerLayer

class CustomBayesianModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=6):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Stack of Bayesian transformer layers
        self.layers = nn.ModuleList([
            BayesianExpectationTransformerLayer({
                'd_model': d_model,
                'n_heads': n_heads,
                'vocab_size': vocab_size,
                'k_permutations': 5,
                'dropout': 0.2
            })
            for _ in range(n_layers)
        ])

        # Output head
        self.output = nn.Linear(d_model, 2)  # Binary classification

    def forward(self, x, return_uncertainty=False):
        # Embed
        x = self.embedding(x)

        # Process through layers
        uncertainties = []
        for layer in self.layers:
            out = layer(x, return_uncertainty=return_uncertainty)
            x = out['hidden_states']

            if return_uncertainty and 'epistemic_uncertainty' in out:
                uncertainties.append(out['epistemic_uncertainty'])

        # Pool and classify
        pooled = x.mean(dim=1)  # Simple mean pooling
        logits = self.output(pooled)

        result = {'logits': logits}

        # Aggregate uncertainties
        if uncertainties:
            # Average uncertainty across layers
            result['uncertainty'] = torch.stack(uncertainties).mean(dim=0)

        return result
```

### Scenario 3: Inference-Only (Load Trained Model)

**Use Case**: You have a trained model and want to deploy for inference.

```python
import torch
from src.bayesian_transformer import BayesianExpectationTransformerLayer

# Load trained model
model = torch.load('path/to/trained_model.pt', map_location='cpu')
model.eval()

# Load calibrator
from src.bayesian_transformer.uncertainty_calibration import PlattScaling
import pickle

with open('path/to/calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)

# Inference function
def predict_with_uncertainty(text_tokens, model, calibrator):
    """
    Predict class and return calibrated uncertainty.

    Args:
        text_tokens: Tokenized input (batch_size, seq_length)
        model: Trained Bayesian model
        calibrator: Fitted calibrator

    Returns:
        predictions: Class predictions
        uncertainties: Calibrated uncertainties
        confidences: Confidence scores
    """
    with torch.no_grad():
        # Forward pass with uncertainty
        output = model(text_tokens, return_uncertainty=True)

        logits = output['logits']
        predictions = logits.argmax(dim=-1)

        # Get raw uncertainty
        raw_uncertainty = output.get('uncertainty', torch.zeros(logits.size(0)))

        # Calibrate uncertainty
        calibrated_uncertainty = calibrator.transform(
            raw_uncertainty.cpu().numpy()
        )

        # Confidence = 1 - uncertainty
        confidence = 1.0 - calibrated_uncertainty

    return predictions, calibrated_uncertainty, confidence


# Example usage
sample_input = torch.randint(0, 10000, (1, 128))  # 1 sample, 128 tokens
pred, unc, conf = predict_with_uncertainty(sample_input, model, calibrator)

print(f"Prediction: {pred.item()}")
print(f"Uncertainty: {unc[0]:.4f}")
print(f"Confidence: {conf[0]:.4f}")

# Decision making
if conf[0] > 0.9:
    print("High confidence - auto-accept")
elif conf[0] > 0.7:
    print("Medium confidence - flag for review")
else:
    print("Low confidence - escalate to human")
```

---

## Performance Optimization

### 1. Reduce Inference Latency

**Problem**: Bayesian layer is 20x slower than standard transformer.

**Solutions**:

```python
# A) Use fewer permutations at inference
config['k_permutations'] = 3  # Reduce from 5 to 3 (-40% time)

# B) Disable uncertainty when not needed
output = model(x, return_uncertainty=False)  # ~10% faster

# C) Use mixed precision (FP16)
from torch.cuda.amp import autocast

with autocast():
    output = model(x)  # ~30% faster on GPU

# D) Batch processing
# Process multiple samples at once (e.g., batch_size=32)

# E) Model compilation (PyTorch 2.0+)
compiled_model = torch.compile(model, mode='reduce-overhead')
```

### 2. Reduce Memory Usage

```python
# A) Use gradient checkpointing during training
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(self, x):
    return checkpoint(self.bayesian_layer, x, use_reentrant=False)

# B) Quantization (INT8)
import torch.quantization as quant

# Quantize model for inference
quantized_model = quant.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# C) Prune permutation matrices
# Keep only top-k permutations (e.g., 3 out of 5)

# D) Use smaller embedding dimension
config['d_model'] = 64  # Instead of 128
```

### 3. Scale to Production

**API Server Example** (FastAPI):

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model at startup
model = torch.load('model.pt')
model.eval()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    uncertainty: float
    confidence: float
    recommendation: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Tokenize (use your tokenizer)
    tokens = tokenize(request.text)

    # Predict
    pred, unc, conf = predict_with_uncertainty(tokens, model, calibrator)

    # Decision logic
    if conf[0] > 0.9:
        recommendation = "auto_accept"
    elif conf[0] > 0.7:
        recommendation = "review"
    else:
        recommendation = "escalate"

    return PredictionResponse(
        prediction=int(pred[0]),
        uncertainty=float(unc[0]),
        confidence=float(conf[0]),
        recommendation=recommendation
    )

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Monitoring & Maintenance

### 1. Track Calibration Drift

```python
from src.bayesian_transformer.uncertainty_calibration import compute_calibration_metrics

# Periodically evaluate calibration on new data
def monitor_calibration(model, calibrator, test_loader):
    """Monitor if calibration is drifting."""

    uncertainties = []
    errors = []

    for batch in test_loader:
        with torch.no_grad():
            output = model(batch['input'], return_uncertainty=True)
            pred = output['logits'].argmax(dim=-1)

            # Collect uncertainties
            raw_unc = output['uncertainty'].cpu().numpy()
            calibrated = calibrator.transform(raw_unc)
            uncertainties.extend(calibrated)

            # Collect errors
            errors.extend((pred != batch['label']).cpu().numpy())

    # Compute ECE
    metrics = compute_calibration_metrics(
        np.array(uncertainties),
        np.array(errors)
    )

    print(f"ECE: {metrics['ECE']:.4f}")
    print(f"MCE: {metrics['MCE']:.4f}")

    # Alert if drift detected
    if metrics['ECE'] > 0.1:
        print("[WARN] Calibration drift detected - consider re-calibration")

    return metrics
```

### 2. A/B Testing

```python
def ab_test_bayesian_vs_standard(test_data):
    """
    Compare Bayesian vs Standard transformer in production.

    Route 50% of traffic to each model, measure:
    - Accuracy
    - Latency
    - False positive/negative rates with/without uncertainty filtering
    """

    results = {
        'bayesian': {'correct': 0, 'total': 0, 'latency': []},
        'standard': {'correct': 0, 'total': 0, 'latency': []}
    }

    for i, sample in enumerate(test_data):
        # Route traffic
        model_type = 'bayesian' if i % 2 == 0 else 'standard'

        # Measure latency
        import time
        start = time.time()

        if model_type == 'bayesian':
            pred, unc, conf = predict_with_uncertainty(sample, bayesian_model, calibrator)
            # Use uncertainty for filtering
            if conf[0] < 0.7:
                continue  # Skip low-confidence predictions
        else:
            pred = standard_model(sample).argmax()

        latency = time.time() - start

        # Track metrics
        results[model_type]['latency'].append(latency)
        results[model_type]['total'] += 1
        if pred == sample['label']:
            results[model_type]['correct'] += 1

    # Report
    for model in ['bayesian', 'standard']:
        acc = results[model]['correct'] / results[model]['total']
        lat = np.mean(results[model]['latency'])
        print(f"{model}: Acc={acc:.4f}, Latency={lat*1000:.2f}ms")
```

### 3. Re-Calibration Trigger

```python
def should_recalibrate(current_ece, baseline_ece=0.05, threshold=0.1):
    """
    Decide if re-calibration is needed.

    Args:
        current_ece: Current ECE on recent data
        baseline_ece: ECE when model was deployed
        threshold: Maximum allowed drift

    Returns:
        bool: True if re-calibration needed
    """
    drift = current_ece - baseline_ece

    if drift > threshold:
        print(f"[ALERT] ECE drift detected: {drift:.4f}")
        print("Recommendation: Re-calibrate model")
        return True

    return False
```

---

## Best Practices

### 1. When to Use Uncertainty

**Always Use** (High Stakes):
- Medical diagnosis
- Financial decisions
- Legal/compliance
- Safety-critical systems

**Selectively Use** (Performance Critical):
- Real-time systems: Disable uncertainty for <1ms latency
- Batch processing: Enable uncertainty for quality filtering

**Never Need** (Low Stakes):
- Spam detection
- Content recommendation
- Simple classification

### 2. Confidence Thresholds

Calibrate thresholds based on your use case:

```python
# Conservative (favor false negatives)
HIGH_CONFIDENCE = 0.95  # Very confident
MEDIUM_CONFIDENCE = 0.85  # Moderately confident

# Balanced
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.75

# Aggressive (favor false positives)
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.60

# Decision logic
if confidence > HIGH_CONFIDENCE:
    auto_accept()
elif confidence > MEDIUM_CONFIDENCE:
    flag_for_review()
else:
    escalate_to_human()
```

### 3. Calibration Maintenance

```python
# Re-calibrate monthly or when drift detected
def recalibration_pipeline(model, new_validation_data):
    """
    Full re-calibration pipeline.

    1. Collect new validation uncertainties
    2. Re-fit calibrator
    3. Validate calibration
    4. Deploy if improved
    """

    # Step 1: Collect data
    uncertainties, errors = collect_validation_data(model, new_validation_data)

    # Step 2: Re-fit
    new_calibrator = PlattScaling()
    new_calibrator.fit(uncertainties, errors)

    # Step 3: Validate
    metrics = compute_calibration_metrics(
        new_calibrator.transform(uncertainties),
        errors
    )

    # Step 4: Deploy if better
    if metrics['ECE'] < current_ece:
        save_calibrator(new_calibrator, 'calibrator_v2.pkl')
        print(f"Deployed new calibrator: ECE {current_ece:.4f} -> {metrics['ECE']:.4f}")
    else:
        print("New calibrator worse - keeping current")
```

---

## Troubleshooting

### Issue 1: Slow Inference

**Symptoms**: >100ms per sample

**Solutions**:
1. Reduce `k_permutations` from 5 to 3
2. Use FP16 mixed precision
3. Disable uncertainty when not needed
4. Use GPU (RTX 3090: ~1ms per sample)

### Issue 2: High Memory Usage

**Symptoms**: OOM errors

**Solutions**:
1. Reduce batch size
2. Use gradient checkpointing
3. Quantize model (INT8)
4. Reduce `d_model` dimension

### Issue 3: Poor Uncertainty Calibration

**Symptoms**: ECE > 0.1

**Solutions**:
1. Collect more calibration data (>5K samples)
2. Try different calibration method (Platt vs Isotonic)
3. Re-train model with uncertainty-aware loss
4. Check for distribution shift

### Issue 4: Low Uncertainty-Error Correlation

**Symptoms**: Correlation < 0.3

**Solutions**:
1. This is OK! Calibration (ECE) is more important than correlation
2. Collect more diverse training data
3. Increase model capacity
4. Use ensemble of multiple models

---

## Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY src/ ./src/
COPY models/ ./models/

# Copy API server
COPY server.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t bayesian-transformer .
docker run -p 8000:8000 bayesian-transformer

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'
```

---

## Summary Checklist

Before deploying to production, verify:

- [ ] Model trained and validated (>85% test accuracy)
- [ ] Calibrator fitted on validation set (ECE < 0.1)
- [ ] Performance benchmarked (latency, memory)
- [ ] API server tested (load testing)
- [ ] Monitoring setup (ECE tracking)
- [ ] A/B test planned (vs baseline)
- [ ] Rollback plan prepared
- [ ] Documentation updated
- [ ] Team trained on interpretation

---

**Status**: Ready for Production ✅
**Support**: See [README.md](../README.md) for issues/questions
**Updates**: Check [PROJECT_STATUS.md](../PROJECT_STATUS.md) for roadmap
