# Component Interaction Diagrams

**Version:** 1.0
**Date:** 2025-10-25
**Related:** [System Architecture](SYSTEM_ARCHITECTURE.md)

---

## 1. Training Workflow Sequence

```
┌─────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│User │     │  Config  │     │ Dataset  │     │  Model   │
└──┬──┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
   │             │                 │                 │
   │ Train       │                 │                 │
   ├────────────▶│                 │                 │
   │             │ Load Dataset    │                 │
   │             ├────────────────▶│                 │
   │             │                 │ Return DataLoader
   │             │◀────────────────┤                 │
   │             │ Initialize Model│                 │
   │             ├────────────────────────────────▶ │
   │             │                 │ Return Model    │
   │             │◀────────────────────────────────┤
   │             │                 │                 │
   │   ┌─────────┴─────────────────┴─────────────────┴──────┐
   │   │            Training Loop                             │
   │   │  ┌─────────────────────────────────────────────┐   │
   │   │  │ For each epoch:                              │   │
   │   │  │   ┌─────────────────────────────────────┐   │   │
   │   │  │   │ For each batch:                     │   │   │
   │   │  │   │   1. Forward pass                   │   │   │
   │   │  │   │   2. Compute loss (MDL regularized) │   │   │
   │   │  │   │   3. Backward pass                  │   │   │
   │   │  │   │   4. Optimizer step                 │   │   │
   │   │  │   │   5. Log metrics                    │   │   │
   │   │  │   └─────────────────────────────────────┘   │   │
   │   │  │   6. Validation                              │   │
   │   │  │   7. Save checkpoint (if improvement)        │   │
   │   │  └─────────────────────────────────────────────┘   │
   │   └──────────────────────────────────────────────────────┘
   │             │                 │                 │
   │             │ Save Checkpoint │                 │
   │             ├────────────────▶│                 │
   │             │                 │ Persist to Disk │
   │             │                 ├────────────────▶│
   │◀────────────┤                 │                 │
   │ Complete    │                 │                 │
   │             │                 │                 │
```

### Key Steps:

1. **Configuration Loading**
   - Load training hyperparameters
   - Initialize random seeds for reproducibility
   - Setup device (CPU/GPU)

2. **Dataset Preparation**
   - Load IMDB dataset via HuggingFace Datasets
   - Tokenize text with caching
   - Create DataLoader with batching

3. **Model Initialization**
   - Instantiate BayesianExpectationTransformerLayer
   - Move to device
   - Setup optimizer (AdamW) and scheduler

4. **Training Loop**
   - Forward pass through model
   - Compute MDL-regularized loss
   - Backward pass for gradients
   - Optimizer step
   - Metrics logging (TensorBoard)

5. **Validation & Checkpointing**
   - Run validation after each epoch
   - Save checkpoint on improvement
   - Early stopping if no improvement

---

## 2. Model Forward Pass Flow

```
┌──────────┐
│  Input   │
│  Tensor  │
│ (B,S,D)  │
└────┬─────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  SufficientStatsEncoder                             │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Compute moments (up to O(log d))       │    │
│  │  2. Compute counting statistics            │    │
│  │  3. Estimate Beta posterior (α, β)         │    │
│  │  4. Return sufficient statistics + posterior│   │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ sufficient_stats (B,S,D)
     ▼
┌─────────────────────────────────────────────────────┐
│  Add & Normalize                                    │
│  x = LayerNorm(x + sufficient_stats)                │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  MartingaleAwareAttention                           │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Standard multi-head attention          │    │
│  │  2. Permutation-based variance reduction   │    │
│  │     - Generate k permutations              │    │
│  │     - Compute attention for each           │    │
│  │     - Average outputs (√k reduction)       │    │
│  │  3. Adaptive weighting (log(n)/n)          │    │
│  │  4. Combine standard + permutation avg     │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ attention_output (B,S,D)
     ▼
┌─────────────────────────────────────────────────────┐
│  Add & Normalize                                    │
│  x = LayerNorm(x + attention_output)                │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  OptimalCoTLayer (Gradient Flow Only)               │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Estimate reasoning entropy (H_CoT)     │    │
│  │  2. Compute optimal length k*              │    │
│  │  3. Generate CoT tokens (if requested)     │    │
│  │  4. Tiny gradient flow contribution        │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ cot_output (metadata only)
     ▼
┌─────────────────────────────────────────────────────┐
│  PositionalDebiasing                                │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Detect periodic artifacts (64, 128,    │    │
│  │     256, 512 token periods)                │    │
│  │  2. Multi-harmonic modeling                │    │
│  │  3. Compute adaptive correction            │    │
│  │  4. Apply position-aware gating            │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ debiased_output (B,S,D)
     ▼
┌─────────────────────────────────────────────────────┐
│  Feed-Forward Network                               │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Linear(D → 4D) + ReLU + Dropout        │    │
│  │  2. Linear(4D → D)                         │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ ffn_output (B,S,D)
     ▼
┌─────────────────────────────────────────────────────┐
│  Add & Normalize                                    │
│  output = LayerNorm(x + ffn_output)                 │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  Return Dictionary                                   │
│  {                                                   │
│    'hidden_states': output (B,S,D),                 │
│    'sufficient_stats': stats_dict,                  │
│    'uncertainty': uncertainty_dict (optional),      │
│    'cot_output': cot_dict,                          │
│    'debiasing_info': debiasing_dict                 │
│  }                                                   │
└──────────────────────────────────────────────────────┘
```

### Tensor Shape Legend:
- B: Batch size
- S: Sequence length
- D: Model dimension (d_model)

---

## 3. Inference API Request Flow

```
┌────────┐
│ Client │
└───┬────┘
    │ POST /predict
    │ {"text": "...", "return_uncertainty": true}
    ▼
┌─────────────────────────────────────────────────────┐
│  FastAPI Endpoint Handler                           │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Receive HTTP request                   │    │
│  │  2. Parse JSON body                        │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Pydantic Validation                                │
│  ┌────────────────────────────────────────────┐    │
│  │  - Validate text (min/max length)          │    │
│  │  - Validate return_uncertainty (bool)      │    │
│  │  - Validate confidence_threshold (0-1)     │    │
│  │  - Return 422 if invalid                   │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ Validated PredictionRequest
     ▼
┌─────────────────────────────────────────────────────┐
│  Tokenizer Pipeline                                 │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Clean text (remove HTML, etc.)         │    │
│  │  2. Tokenize with BPE/WordPiece            │    │
│  │  3. Add special tokens ([CLS], [SEP])      │    │
│  │  4. Pad/truncate to max_length             │    │
│  │  5. Create attention mask                  │    │
│  │  6. Convert to tensors                     │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ input_ids (1,S), attention_mask (1,S)
     ▼
┌─────────────────────────────────────────────────────┐
│  Model Inference                                    │
│  ┌────────────────────────────────────────────┐    │
│  │  with torch.no_grad():                     │    │
│  │    1. Load model from cache (if needed)    │    │
│  │    2. Move inputs to device (GPU/CPU)      │    │
│  │    3. Forward pass                         │    │
│  │    4. Get hidden_states + uncertainty      │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ model_output dict
     ▼
┌─────────────────────────────────────────────────────┐
│  Postprocessing                                     │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Extract [CLS] token representation     │    │
│  │  2. Apply classification head              │    │
│  │  3. Compute softmax probabilities          │    │
│  │  4. Get predicted class                    │    │
│  │  5. Extract uncertainty metrics            │    │
│  │  6. Check confidence threshold             │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Response Formatting                                │
│  ┌────────────────────────────────────────────┐    │
│  │  Build PredictionResponse:                 │    │
│  │  {                                         │    │
│  │    "prediction": "positive",               │    │
│  │    "confidence": 0.92,                     │    │
│  │    "uncertainty": {                        │    │
│  │      "epistemic": 0.05,                    │    │
│  │      "aleatoric": 0.03,                    │    │
│  │      "total": 0.08                         │    │
│  │    },                                      │    │
│  │    "cot_length": null,                     │    │
│  │    "reasoning": null                       │    │
│  │  }                                         │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ JSON response
     ▼
┌────────┐
│ Client │ ← HTTP 200 + JSON
└────────┘
```

### Error Handling Flow:

```
┌─────────────────┐
│  Error Occurs   │
└────┬────────────┘
     │
     ├─ ValueError (invalid input)
     │   └─▶ HTTP 422 Unprocessable Entity
     │       {"detail": "Text too long"}
     │
     ├─ ModelNotFoundError
     │   └─▶ HTTP 503 Service Unavailable
     │       {"detail": "Model not loaded"}
     │
     ├─ LowConfidenceError
     │   └─▶ HTTP 422 Unprocessable Entity
     │       {"detail": "Confidence below threshold"}
     │
     └─ Exception (unexpected)
         └─▶ HTTP 500 Internal Server Error
             {"detail": "Internal error"}
             + Log to error tracker
```

---

## 4. Checkpoint Save/Load Flow

### Save Checkpoint

```
┌──────────────┐
│Training Loop │
└──────┬───────┘
       │ epoch_end or validation_improvement
       ▼
┌─────────────────────────────────────────────────────┐
│  CheckpointManager.save_training_checkpoint()       │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Gather checkpoint data:                │    │
│  │     - model.state_dict()                   │    │
│  │     - optimizer.state_dict()               │    │
│  │     - scheduler.state_dict()               │    │
│  │     - epoch, global_step, metrics          │    │
│  │     - config, metadata                     │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Build Checkpoint Dictionary                        │
│  {                                                  │
│    'version': '1.0.0',                              │
│    'timestamp': datetime.now(),                     │
│    'model_state_dict': OrderedDict(...),            │
│    'optimizer_state_dict': {...},                   │
│    'epoch': 10,                                     │
│    'metrics': {'loss': 0.123, 'accuracy': 0.92},    │
│    'config': {...}                                  │
│  }                                                  │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Atomic Write                                       │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Generate filename:                     │    │
│  │     epoch_0010_step_12345.pt               │    │
│  │  2. Write to temp file:                    │    │
│  │     epoch_0010_step_12345.pt.tmp           │    │
│  │  3. Atomic rename (temp → final)           │    │
│  │  4. Update 'latest' symlink                │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Cleanup Old Checkpoints                            │
│  ┌────────────────────────────────────────────┐    │
│  │  1. List all training checkpoints          │    │
│  │  2. Sort by modification time              │    │
│  │  3. Keep last K checkpoints (default: 3)   │    │
│  │  4. Delete older checkpoints               │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│  Continue    │
│  Training    │
└──────────────┘
```

### Load Checkpoint

```
┌──────────────┐
│ User Request │
│ Resume Train │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  CheckpointManager.load_checkpoint()                │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Find checkpoint file:                  │    │
│  │     - 'latest.pt' (symlink to most recent) │    │
│  │     - Or specific path provided            │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Load & Validate                                    │
│  ┌────────────────────────────────────────────┐    │
│  │  1. Load checkpoint from disk              │    │
│  │  2. Validate structure (required keys)     │    │
│  │  3. Check version compatibility            │    │
│  │  4. Return checkpoint dictionary           │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ checkpoint dict
     ▼
┌─────────────────────────────────────────────────────┐
│  Restore State                                      │
│  ┌────────────────────────────────────────────┐    │
│  │  1. model.load_state_dict(...)             │    │
│  │  2. optimizer.load_state_dict(...)         │    │
│  │  3. scheduler.load_state_dict(...)         │    │
│  │  4. Set epoch, global_step                 │    │
│  │  5. Restore metrics history                │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│  Continue    │
│  Training    │
└──────────────┘
```

---

## 5. Data Pipeline Flow

```
┌──────────────┐
│  IMDB Raw    │
│  Dataset     │
└──────┬───────┘
       │ load_dataset("imdb")
       ▼
┌─────────────────────────────────────────────────────┐
│  HuggingFace Dataset Object                         │
│  {                                                  │
│    'train': Dataset(...),                           │
│    'test': Dataset(...)                             │
│  }                                                  │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Tokenization (map with batching)                   │
│  ┌────────────────────────────────────────────┐    │
│  │  def tokenize_function(examples):          │    │
│  │    return tokenizer(                       │    │
│  │      examples['text'],                     │    │
│  │      padding='max_length',                 │    │
│  │      truncation=True,                      │    │
│  │      max_length=512                        │    │
│  │    )                                       │    │
│  │                                            │    │
│  │  tokenized = dataset.map(                  │    │
│  │    tokenize_function,                      │    │
│  │    batched=True,                           │    │
│  │    num_proc=4  # parallel processing       │    │
│  │  )                                         │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │ Tokenized Dataset (cached on disk)
     ▼
┌─────────────────────────────────────────────────────┐
│  Format for PyTorch                                 │
│  ┌────────────────────────────────────────────┐    │
│  │  tokenized.set_format(                     │    │
│  │    type='torch',                           │    │
│  │    columns=['input_ids',                   │    │
│  │             'attention_mask',               │    │
│  │             'label']                        │    │
│  │  )                                         │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Create DataLoader                                  │
│  ┌────────────────────────────────────────────┐    │
│  │  DataLoader(                               │    │
│  │    tokenized,                              │    │
│  │    batch_size=32,                          │    │
│  │    shuffle=True,                           │    │
│  │    num_workers=4,                          │    │
│  │    pin_memory=True,  # GPU optimization    │    │
│  │    prefetch_factor=2  # preload batches    │    │
│  │  )                                         │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Batch Iterator                                     │
│  ┌────────────────────────────────────────────┐    │
│  │  for batch in dataloader:                  │    │
│  │    {                                       │    │
│  │      'input_ids': Tensor(32, 512),         │    │
│  │      'attention_mask': Tensor(32, 512),    │    │
│  │      'label': Tensor(32,)                  │    │
│  │    }                                       │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│ Model Input  │
└──────────────┘
```

### Optimization Techniques:

1. **Caching:** Tokenized dataset cached on disk (no re-tokenization)
2. **Parallel Processing:** num_proc=4 for tokenization
3. **Prefetching:** prefetch_factor=2 overlaps data loading with training
4. **Pin Memory:** pin_memory=True for faster GPU transfer
5. **Batching:** Process multiple samples simultaneously

---

## 6. Monitoring & Metrics Collection

```
┌──────────────┐
│Training Step │
└──────┬───────┘
       │ Every step/epoch
       ▼
┌─────────────────────────────────────────────────────┐
│  Metrics Collection                                 │
│  ┌────────────────────────────────────────────┐    │
│  │  Training Metrics:                         │    │
│  │  - loss (total, standard, mdl_penalty)     │    │
│  │  - accuracy, precision, recall, f1         │    │
│  │  - learning_rate                           │    │
│  │  - gradient_norm, parameter_norm           │    │
│  │                                            │    │
│  │  Model Metrics:                            │    │
│  │  - optimal_cot_length (mean, std)          │    │
│  │  - reasoning_entropy                       │    │
│  │  - uncertainty (epistemic, aleatoric)      │    │
│  │  - martingale_violations                   │    │
│  │                                            │    │
│  │  System Metrics:                           │    │
│  │  - gpu_utilization, gpu_memory             │    │
│  │  - throughput (samples/sec)                │    │
│  │  - time_per_epoch                          │    │
│  └────────────────────────────────────────────┘    │
└────┬────────────────────────────────────────────────┘
     │
     ├─────────────────────────────────────┐
     │                                     │
     ▼                                     ▼
┌──────────────┐                  ┌──────────────┐
│ TensorBoard  │                  │ Weights &    │
│  Logger      │                  │  Biases      │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       │ Write to logs/                  │ Log to W&B API
       │   - scalar metrics              │   - metrics
       │   - histograms                  │   - charts
       │   - model graph                 │   - artifacts
       │                                 │
       ▼                                 ▼
┌──────────────┐                  ┌──────────────┐
│TensorBoard UI│                  │  W&B UI      │
│localhost:6006│                  │  wandb.ai    │
└──────────────┘                  └──────────────┘
```

### Metrics Dashboard Layout:

**Training Tab:**
- Loss curves (total, standard, MDL penalty)
- Accuracy/F1 over time
- Learning rate schedule
- Gradient norms

**Model Tab:**
- Optimal CoT length distribution
- Reasoning entropy heatmap
- Uncertainty calibration plot
- Martingale violation convergence

**System Tab:**
- GPU utilization %
- GPU memory usage
- Training throughput
- Epoch timing

---

## 7. Error Handling & Recovery

```
┌──────────────┐
│ System Event │
└──────┬───────┘
       │
       ├─ GPU Out of Memory (OOM)
       │   ├─▶ Reduce batch size
       │   ├─▶ Enable gradient checkpointing
       │   └─▶ Retry training
       │
       ├─ Training Interruption
       │   ├─▶ Save emergency checkpoint
       │   ├─▶ Log interruption reason
       │   └─▶ Exit gracefully
       │
       ├─ Validation Loss Spike
       │   ├─▶ Load best checkpoint
       │   ├─▶ Reduce learning rate
       │   └─▶ Continue training
       │
       ├─ Data Loading Error
       │   ├─▶ Skip corrupted batch
       │   ├─▶ Log error details
       │   └─▶ Continue with next batch
       │
       └─ Model NaN/Inf Values
           ├─▶ Log problematic batch
           ├─▶ Load last valid checkpoint
           ├─▶ Apply gradient clipping
           └─▶ Retry with lower learning rate
```

---

## Summary

These component interaction diagrams provide detailed views of:

1. **Training Workflow:** End-to-end training process
2. **Model Forward Pass:** Internal tensor flow through all components
3. **Inference API:** Request handling from client to response
4. **Checkpointing:** Save/load mechanisms for fault tolerance
5. **Data Pipeline:** IMDB dataset preprocessing and batching
6. **Monitoring:** Metrics collection and visualization
7. **Error Handling:** Fault tolerance and recovery strategies

All diagrams are designed to be:
- **Clear:** Easy to understand flows
- **Comprehensive:** Cover all major interactions
- **Actionable:** Guide implementation decisions
- **Maintainable:** Easy to update as system evolves

---

**Last Updated:** 2025-10-25
**Related Documents:**
- [System Architecture](SYSTEM_ARCHITECTURE.md)
- [ADR-001: PyTorch Framework](../adrs/ADR-001-pytorch-over-tensorflow.md)
- [ADR-002: FastAPI](../adrs/ADR-002-fastapi-for-rest-api.md)
- [ADR-004: Checkpointing](../adrs/ADR-004-model-checkpointing-strategy.md)
