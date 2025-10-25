# Quick Reference: ML Deployment Best Practices 2025

## 🚀 Fastest Wins (Implement First)

### 1. Data Loading (2-3x Speedup)
```python
# Enable streaming + parallel workers
dataset = load_dataset("stanfordnlp/imdb", streaming=True)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

### 2. Model Checkpointing (10x Memory Reduction)
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self._forward_impl, x, use_reentrant=False)
```

### 3. FastAPI Model Loading (100x Speedup)
```python
# Load at startup, NOT per request
@app.on_event("startup")
async def load_model():
    global model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
```

### 4. Redis Caching (60% Load Reduction)
```python
cached = redis_client.get(cache_key)
if cached:
    return cached
result = model.predict(text)
redis_client.setex(cache_key, 3600, result)
```

---

## 📊 Critical Code Snippets

### Production Checkpoint Manager
```python
class CheckpointManager:
    def save_checkpoint(self, model, optimizer, scheduler, epoch, step, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'torch_rng_state': torch.get_rng_state(),
        }
        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")
```

### TensorBoard Logger
```python
from torch.utils.tensorboard import SummaryWriter

logger = SummaryWriter("C:/dev/coding/Bayesian/runs")
logger.add_scalar('Loss/train', loss, step)
logger.add_histogram('Gradients/layer1', param.grad, step)
```

### HuggingFace Hub Upload
```python
model.push_to_hub(
    "username/bayesian-transformer-imdb",
    commit_message="v1.0 release",
    private=False
)
```

### FastAPI Batch Inference
```python
@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        process_pool,
        batch_inference_sync,
        texts
    )
    return results
```

### Performance Benchmark
```python
class PerformanceBenchmark:
    def benchmark_latency(self, texts, num_runs=100):
        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            latencies.append(time.perf_counter() - start)
        return np.percentile(latencies, [50, 95, 99])
```

---

## 🎯 Production Checklist

### Before Training
- [ ] Enable dataset streaming for large datasets
- [ ] Configure activation checkpointing on transformer blocks
- [ ] Setup TensorBoard logging directory
- [ ] Implement checkpoint manager with max_keep=3
- [ ] Enable mixed precision training (FP16/BF16)
- [ ] Configure FSDP if using multi-GPU

### During Training
- [ ] Monitor gradient norms every 100 steps
- [ ] Save checkpoints every epoch
- [ ] Log attention entropy for collapse detection
- [ ] Track learning rate schedule
- [ ] Validate calibration on held-out set

### After Training
- [ ] Benchmark latency (P50/P95/P99)
- [ ] Measure throughput at different batch sizes
- [ ] Calculate Expected Calibration Error
- [ ] Create comprehensive model card
- [ ] Upload to HuggingFace Hub with tags
- [ ] Include TensorBoard logs in repo

### API Deployment
- [ ] Load model once at startup
- [ ] Implement ProcessPoolExecutor for batching
- [ ] Add Redis caching layer
- [ ] Configure rate limiting (100/min)
- [ ] Setup health check endpoint
- [ ] Enable Prometheus metrics
- [ ] Deploy with Docker + 4 Uvicorn workers
- [ ] Test with load testing tool (locust/k6)

---

## ⚡ Performance Numbers (2025)

| Optimization | Impact |
|---|---|
| Activation checkpointing | 10x memory reduction, 10-20% slowdown |
| Streaming datasets | Instant start vs full download |
| Parallel DataLoader (4 workers) | 2-3x speedup |
| Model loading at startup | 100x faster than per-request |
| Redis caching | 60% load reduction |
| torch.compile() | 20-30% inference speedup |
| FP8 quantization (TorchAO) | 17% throughput gain |
| FSDP2 | Train models > 1B params on multi-GPU |
| Batch size 32 vs 1 | 5-10x throughput increase |

---

## 🔧 Common Issues & Solutions

### Out of Memory (OOM)
```python
# Solution 1: Activation checkpointing
x = checkpoint(layer, x, use_reentrant=False)

# Solution 2: Gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(**batch)
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Reduce batch size
batch_size = 16  # Instead of 32
```

### Slow Training
```python
# Enable mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**inputs)
    loss = outputs.loss
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Use parallel data loading
dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)
```

### Poor Calibration
```python
# Temperature scaling
def temperature_scale(logits, temperature=1.5):
    return logits / temperature

# Apply after training
logits = model(**inputs).logits
calibrated_probs = torch.softmax(temperature_scale(logits), dim=-1)
```

### Slow API Response
```python
# Add caching
@lru_cache(maxsize=1000)
def cached_inference(text_hash):
    return model.predict(text)

# Use batch processing
@app.post("/batch_predict")
async def batch_predict(texts: List[str]):
    # Process all at once
    return await batch_inference_async(texts)
```

---

## 📚 Key Libraries & Versions (2025)

```toml
[dependencies]
torch = "^2.5.0"
transformers = "^4.50.0"
datasets = "^2.21.0"
fastapi = "^0.109.0"
uvicorn = "^0.27.0"
tensorboard = "^2.16.0"
redis = "^5.0.0"
celery = "^5.3.0"
prometheus-client = "^0.19.0"
```

---

## 🎓 Essential Reading

1. **HuggingFace Datasets Streaming**: https://huggingface.co/docs/datasets/stream
2. **PyTorch FSDP2 Tutorial**: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
3. **FastAPI ML Serving**: https://fastapi.tiangolo.com/
4. **TensorBoard with PyTorch**: https://docs.pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
5. **HuggingFace Hub Publishing**: https://huggingface.co/docs/hub/models-uploading

---

## 🚨 Critical Don'ts

❌ **Never** load models in request handlers (use startup event)
❌ **Never** skip checkpointing optimizer/scheduler state
❌ **Never** ignore P95/P99 latency (mean is misleading)
❌ **Never** deploy without health checks
❌ **Never** forget to validate calibration (accuracy isn't enough)
❌ **Never** use synchronous inference in async FastAPI
❌ **Never** cache without expiration (memory leaks)
❌ **Never** skip model card documentation

---

**Last Updated:** October 25, 2025
**Maintained by:** Bayesian Transformer Research Team
