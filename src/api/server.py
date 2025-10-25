from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import sys

# Add project root to Python path (for uvicorn)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.bayesian_transformer import BayesianExpectationTransformerLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model (loaded at startup)
model: Optional[BayesianExpectationTransformerLayer] = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global model

    logger.info("Loading model...")

    # Load model from checkpoint
    checkpoint_path = Path('checkpoints/production/latest.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']

        model = BayesianExpectationTransformerLayer(**config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        logger.info(f"Model loaded from {checkpoint_path}")
    else:
        # Initialize with default config for demo
        default_config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 10000,
            'dropout': 0.1,
            'k_permutations': 20,
            'epsilon': 0.05
        }
        model = BayesianExpectationTransformerLayer(default_config).to(device)
        model.eval()
        logger.info("Model initialized with default config")

    yield

    # Cleanup
    logger.info("Shutting down...")
    del model

app = FastAPI(
    title="Bayesian Transformer API",
    description="REST API for Bayesian Expectation Transformer with uncertainty quantification",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Input text to classify", min_length=1)
    return_uncertainty: bool = Field(
        default=True,
        description="Whether to return uncertainty metrics"
    )
    temperature: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature"
    )

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted class (0-1)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    epistemic_uncertainty: Optional[float] = Field(
        None,
        description="Epistemic (model) uncertainty"
    )
    aleatoric_uncertainty: Optional[float] = Field(
        None,
        description="Aleatoric (data) uncertainty"
    )
    should_route_to_human: bool = Field(
        ...,
        description="Whether to route to human review (high uncertainty)"
    )

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify")
    return_uncertainty: bool = Field(default=True)
    batch_size: int = Field(default=8, ge=1, le=64)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str

class ModelInfoResponse(BaseModel):
    config: Dict[str, Any]
    num_parameters: int
    device: str
    metrics: Optional[Dict[str, float]] = None

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        version="1.0.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model configuration and metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    num_params = sum(p.numel() for p in model.parameters())

    config = {
        'd_model': model.d_model,
        'n_heads': model.attention.n_heads,
        'k_permutations': model.attention.k_permutations,
    }

    return ModelInfoResponse(
        config=config,
        num_parameters=num_params,
        device=str(device)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single text prediction with uncertainty quantification."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Simple tokenization (for demo)
        # In production, use proper tokenizer
        input_ids = torch.randint(0, 10000, (1, 128)).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_ids, return_uncertainty=request.return_uncertainty)

        # Extract results
        prediction = outputs['predictions'][0].mean().item()

        # Calculate confidence from logits variance
        logits_std = outputs['predictions'][0].std().item()
        confidence = 1.0 / (1.0 + logits_std)

        epistemic = None
        aleatoric = None
        if request.return_uncertainty and 'epistemic_uncertainty' in outputs:
            epistemic = outputs['epistemic_uncertainty'][0].item()
            aleatoric = outputs['aleatoric_uncertainty'][0].item()

        # Uncertainty-based routing
        uncertainty_threshold = 0.3
        should_route = (epistemic or 0) > uncertainty_threshold if epistemic else False

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            should_route_to_human=should_route
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction for multiple texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []

        # Process in batches
        for i in range(0, len(request.texts), request.batch_size):
            batch_texts = request.texts[i:i + request.batch_size]

            # Simple tokenization (for demo)
            batch_size = len(batch_texts)
            input_ids = torch.randint(0, 10000, (batch_size, 128)).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_ids, return_uncertainty=request.return_uncertainty)

            # Collect results
            for j in range(len(batch_texts)):
                prediction = outputs['predictions'][j].mean().item()
                logits_std = outputs['predictions'][j].std().item()
                confidence = 1.0 / (1.0 + logits_std)

                result = {
                    'text': batch_texts[j],
                    'prediction': prediction,
                    'confidence': confidence
                }

                if request.return_uncertainty and 'epistemic_uncertainty' in outputs:
                    result['epistemic_uncertainty'] = outputs['epistemic_uncertainty'][j].item()
                    result['aleatoric_uncertainty'] = outputs['aleatoric_uncertainty'][j].item()

                results.append(result)

        return {'predictions': results, 'total': len(results)}

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
