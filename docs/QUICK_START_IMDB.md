# Quick Start: Real IMDB Dataset Integration

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

This will install:
- `torch>=2.0.0`
- `transformers>=4.20.0`
- `datasets>=2.0.0` (NEW - for IMDB data)
- Other dependencies

## Usage

### 1. Run the Demo

```bash
python examples/real_data_demo.py
```

This will:
1. Load 1,000 real IMDB training reviews
2. Load 200 real IMDB test reviews
3. Build vocabulary from real movie reviews
4. Train Bayesian sentiment classifier
5. Show predictions with uncertainty estimates
6. Demonstrate active learning

### 2. Run Tests

```bash
# Simple integration test (no dependencies)
python tests/test_imdb_simple.py
```

Expected output:
```
======================================================================
IMDB DATASET INTEGRATION TEST
======================================================================
Testing IMDB dataset loading from HuggingFace...
Loaded 5 samples
[SUCCESS] IMDB dataset integration working!
...
ALL TESTS PASSED [SUCCESS]
======================================================================
```

## Code Examples

### Load IMDB Data

```python
from datasets import load_dataset

# Load real IMDB reviews
dataset = load_dataset('stanfordnlp/imdb', split='train', streaming=True)

# Get samples
samples = list(dataset.take(100))

# Each sample has:
# - 'text': The movie review (string)
# - 'label': 0 (negative) or 1 (positive)
```

### Create DataLoader

```python
from examples.real_data_demo import load_imdb_data, SimpleTokenizer, create_dataloader

# Load data
train_data = load_imdb_data(split='train', max_samples=1000)

# Create tokenizer
tokenizer = SimpleTokenizer(vocab_size=10000, max_length=128)
train_texts = [item['text'] for item in train_data]
tokenizer.build_vocab(train_texts)

# Create dataloader
train_loader = create_dataloader(
    train_data,
    tokenizer,
    batch_size=32,
    max_length=128
)

# Use in training
for batch in train_loader:
    input_ids = batch['input_ids']  # Shape: (batch_size, seq_len)
    labels = batch['label']          # Shape: (batch_size,)
    texts = batch['text']            # List of strings
    # ... train model
```

### Train Model

```python
from examples.real_data_demo import BayesianSentimentClassifier

# Initialize model
model = BayesianSentimentClassifier(
    vocab_size=len(tokenizer.vocab),
    d_model=128,
    n_heads=4
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(5):
    for batch in train_loader:
        # Forward pass
        outputs = model(batch['input_ids'])
        loss = criterion(outputs['logits'], batch['label'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Get Predictions with Uncertainty

```python
# Evaluate with uncertainty
model.eval()
with torch.no_grad():
    outputs = model(input_ids, return_uncertainty=True)

    # Predictions
    logits = outputs['logits']
    predictions = logits.argmax(dim=-1)

    # Uncertainty
    uncertainty = outputs['epistemic_uncertainty']

    # Filter by confidence
    confident_mask = uncertainty < 0.15
    confident_predictions = predictions[confident_mask]
```

## Configuration Options

### Dataset Size

```python
# Small test (fast)
train_data = load_imdb_data(split='train', max_samples=100)

# Medium (1K samples, ~1 min)
train_data = load_imdb_data(split='train', max_samples=1000)

# Large (5K samples, ~5 min)
train_data = load_imdb_data(split='train', max_samples=5000)

# Full dataset (25K samples, ~30 min)
train_data = load_imdb_data(split='train', streaming=False, max_samples=25000)
```

### Tokenizer Settings

```python
# Small vocabulary (fast, less accurate)
tokenizer = SimpleTokenizer(vocab_size=1000, max_length=64)

# Medium vocabulary (balanced)
tokenizer = SimpleTokenizer(vocab_size=10000, max_length=128)

# Large vocabulary (slower, more accurate)
tokenizer = SimpleTokenizer(vocab_size=30000, max_length=256)
```

### Training Settings

```python
# Quick test
n_epochs = 2
batch_size = 64
learning_rate = 1e-3

# Production
n_epochs = 10
batch_size = 32
learning_rate = 5e-4
```

## Expected Results

### Small Dataset (1K train, 200 test)
- Training time: ~2-3 minutes (CPU)
- Accuracy: ~70-75%
- Vocabulary: ~2,000 words

### Medium Dataset (5K train, 1K test)
- Training time: ~10-15 minutes (CPU)
- Accuracy: ~75-80%
- Vocabulary: ~5,000 words

### Full Dataset (25K train, 25K test)
- Training time: ~30-60 minutes (CPU)
- Accuracy: ~85-90%
- Vocabulary: ~10,000+ words

## Troubleshooting

### Issue: Import Error

```
ImportError: cannot import name 'datasets'
```

**Solution**:
```bash
pip install datasets>=2.0.0
```

### Issue: Slow Loading

**Solution**: Use streaming mode:
```python
dataset = load_imdb_data(split='train', streaming=True, max_samples=1000)
```

### Issue: Out of Memory

**Solution**: Reduce batch size or dataset size:
```python
# Smaller batch
train_loader = create_dataloader(dataset, tokenizer, batch_size=16)

# Fewer samples
dataset = load_imdb_data(max_samples=500)
```

## What's Next?

1. **Experiment**: Try different hyperparameters
2. **Scale Up**: Train on full 25K dataset
3. **Fine-tune**: Adjust uncertainty thresholds
4. **Deploy**: Build REST API for production
5. **Enhance**: Add active learning loop

## Documentation

- Full summary: `docs/IMDB_INTEGRATION_SUMMARY.md`
- Before/After comparison: `docs/BEFORE_AFTER_COMPARISON.md`
- Research reference: `docs/research/ml-deployment-research-2025.md`

## Support

For issues or questions:
1. Check the test file: `tests/test_imdb_simple.py`
2. Review the demo: `examples/real_data_demo.py`
3. Read HuggingFace docs: https://huggingface.co/datasets/stanfordnlp/imdb
