# Bayesian Transformer API Documentation

## Overview

This API provides a production-ready REST endpoint for the Bayesian Expectation Transformer model with uncertainty quantification. The API follows FastAPI best practices from 2025, including load-at-startup pattern for 100x speedup and async processing.

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /health

Health check endpoint for load balancers.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

**Fields:**
- `status`: "healthy" | "unhealthy"
- `model_loaded`: boolean - whether model is loaded in memory
- `device`: string - "cpu" | "cuda"
- `version`: string - API version

---

### GET /model-info

Get model configuration and metadata.

**Response:**

```json
{
  "config": {
    "d_model": 512,
    "n_heads": 8,
    "k_permutations": 5
  },
  "num_parameters": 12500000,
  "device": "cuda",
  "metrics": null
}
```

**Fields:**
- `config`: object - model hyperparameters
- `num_parameters`: int - total trainable parameters
- `device`: string - device where model is loaded
- `metrics`: object (optional) - performance metrics

---

### POST /predict

Single text classification with uncertainty quantification.

**Request:**

```json
{
  "text": "This movie was amazing!",
  "return_uncertainty": true,
  "temperature": 1.0
}
```

**Request Fields:**
- `text`: string (required, min_length=1) - Input text to classify
- `return_uncertainty`: boolean (default: true) - Whether to return uncertainty metrics
- `temperature`: float (default: 1.0, range: 0.1-2.0) - Sampling temperature

**Response:**

```json
{
  "prediction": 0.92,
  "confidence": 0.88,
  "epistemic_uncertainty": 0.15,
  "aleatoric_uncertainty": 0.08,
  "should_route_to_human": false
}
```

**Response Fields:**
- `prediction`: float (0-1) - Predicted class probability
- `confidence`: float (0-1) - Model confidence in prediction
- `epistemic_uncertainty`: float (optional) - Model uncertainty (reducible with more data)
- `aleatoric_uncertainty`: float (optional) - Data uncertainty (irreducible)
- `should_route_to_human`: boolean - Whether to route to human review (high uncertainty)

**Example cURL:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This movie was amazing!",
    "return_uncertainty": true,
    "temperature": 1.0
  }'
```

---

### POST /batch-predict

Batch prediction for multiple texts.

**Request:**

```json
{
  "texts": [
    "Great movie!",
    "Terrible film.",
    "Average experience."
  ],
  "return_uncertainty": true,
  "batch_size": 8
}
```

**Request Fields:**
- `texts`: List[string] (required, max_items=32) - List of texts to classify
- `return_uncertainty`: boolean (default: true) - Whether to return uncertainty metrics
- `batch_size`: int (default: 8, range: 1-64) - Batch size for processing

**Response:**

```json
{
  "predictions": [
    {
      "text": "Great movie!",
      "prediction": 0.95,
      "confidence": 0.92,
      "epistemic_uncertainty": 0.10,
      "aleatoric_uncertainty": 0.05
    },
    {
      "text": "Terrible film.",
      "prediction": 0.05,
      "confidence": 0.89,
      "epistemic_uncertainty": 0.12,
      "aleatoric_uncertainty": 0.06
    },
    {
      "text": "Average experience.",
      "prediction": 0.52,
      "confidence": 0.65,
      "epistemic_uncertainty": 0.28,
      "aleatoric_uncertainty": 0.15
    }
  ],
  "total": 3
}
```

**Response Fields:**
- `predictions`: List[object] - List of prediction results
- `total`: int - Total number of predictions

**Example cURL:**

```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Great movie!", "Terrible film."],
    "return_uncertainty": true,
    "batch_size": 8
  }'
```

---

## Usage Examples

### Python with requests

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'text': 'This is an amazing film!',
        'return_uncertainty': True
    }
)
result = response.json()
print(f"Prediction: {result['prediction']:.2f}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Epistemic Uncertainty: {result['epistemic_uncertainty']:.2f}")

# Batch prediction
response = requests.post(
    'http://localhost:8000/batch-predict',
    json={
        'texts': [
            'Great movie!',
            'Terrible waste of time.',
            'Pretty good overall.'
        ],
        'return_uncertainty': True,
        'batch_size': 8
    }
)
results = response.json()
for pred in results['predictions']:
    print(f"{pred['text']}: {pred['prediction']:.2f} (confidence: {pred['confidence']:.2f})")
```

### Python with httpx (async)

```python
import httpx
import asyncio

async def predict_async(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8000/predict',
            json={'text': text, 'return_uncertainty': True}
        )
        return response.json()

# Run async prediction
result = asyncio.run(predict_async('Amazing movie!'))
print(result)
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');

async function predict(text) {
  const response = await axios.post('http://localhost:8000/predict', {
    text: text,
    return_uncertainty: true,
    temperature: 1.0
  });

  return response.data;
}

predict('This movie was fantastic!')
  .then(result => {
    console.log(`Prediction: ${result.prediction}`);
    console.log(`Confidence: ${result.confidence}`);
    console.log(`Should route to human: ${result.should_route_to_human}`);
  });
```

---

## Error Handling

### 500 Internal Server Error

```json
{
  "detail": "Prediction error: model inference failed"
}
```

### 503 Service Unavailable

```json
{
  "detail": "Model not loaded"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Running the Server

### Development Mode

```bash
# Using the provided script
bash scripts/run_server.sh

# Or directly with uvicorn
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
# Using Docker
docker build -t bayesian-transformer-api .
docker run -p 8000:8000 bayesian-transformer-api

# Or with multiple workers
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Performance Characteristics

Based on FastAPI best practices (2025):

- **Model Load Time**: ~2-5 seconds at startup (loaded once)
- **Inference Latency**: ~10-50ms per request (depending on hardware)
- **Batch Processing**: 2-4x throughput improvement for batch sizes 8-32
- **Memory Usage**: Model loaded in memory (no per-request loading = 100x speedup)

---

## Interactive Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Uncertainty-Based Routing

The API includes intelligent routing based on uncertainty:

- **High Epistemic Uncertainty** (> 0.3): Routes to human review
- **Low Confidence** (< 0.5): Consider alternative models
- **High Aleatoric Uncertainty**: Inherent data ambiguity

Use `should_route_to_human` field for production workflows.

---

## Best Practices

1. **Use batch endpoints** for multiple predictions (higher throughput)
2. **Cache frequent queries** with Redis for production
3. **Monitor /health endpoint** for load balancer health checks
4. **Set appropriate temperature** for calibrated predictions
5. **Handle uncertainty** for critical decision-making applications
6. **Implement rate limiting** for production deployments
7. **Use async clients** (httpx, aiohttp) for concurrent requests

---

## Support

For issues or questions:
- GitHub: https://github.com/yourusername/bayesian-transformer
- Documentation: See `docs/` directory
