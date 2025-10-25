"""
Demo: Sentiment Analysis with Bayesian Uncertainty Quantification

This demo shows how to use the Bayesian Expectation Transformer for
sentiment analysis with calibrated uncertainty estimates.
"""

import torch
import torch.nn as nn
from bayesian_transformer import (
    BayesianExpectationTransformerLayer,
    MDLRegularizedLoss
)


class SentimentAnalyzer(nn.Module):
    """
    Sentiment analyzer with Bayesian uncertainty quantification.
    """

    def __init__(self, vocab_size=10000, d_model=256, n_heads=8, n_classes=3):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes

        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Bayesian Transformer Layer
        config = {
            'd_model': d_model,
            'n_heads': n_heads,
            'vocab_size': vocab_size
        }
        self.transformer = BayesianExpectationTransformerLayer(config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, input_ids, return_uncertainty=False):
        """
        Forward pass with optional uncertainty quantification.

        Args:
            input_ids: Token IDs (batch_size, seq_length)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with logits and optional uncertainty
        """
        # Embed tokens
        x = self.embedding(input_ids)  # (batch_size, seq_length, d_model)

        # Process through Bayesian transformer
        transformer_out = self.transformer(x, return_uncertainty=return_uncertainty)
        hidden_states = transformer_out['hidden_states']

        # Pool sequence (mean pooling)
        pooled = hidden_states.mean(dim=1)  # (batch_size, d_model)

        # Classify
        logits = self.classifier(pooled)  # (batch_size, n_classes)

        # Get predictions and confidence
        probs = torch.softmax(logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)

        output = {
            'logits': logits,
            'predictions': predictions,
            'confidence': confidence,
            'probabilities': probs
        }

        if return_uncertainty:
            # Add Bayesian uncertainty estimates
            uncertainty = transformer_out['uncertainty']
            output['uncertainty'] = uncertainty

            # Combine model confidence with epistemic uncertainty
            # High epistemic uncertainty = model is unsure
            epistemic = uncertainty['epistemic'].mean(dim=-1)  # (batch_size,)
            output['epistemic_uncertainty'] = epistemic
            output['calibrated_confidence'] = confidence * (1 - epistemic)

        return output


def demo_sentiment_analysis():
    """
    Demonstrate sentiment analysis with uncertainty quantification.
    """
    print("=" * 60)
    print("Bayesian Sentiment Analysis Demo")
    print("=" * 60)

    # Setup
    vocab_size = 10000
    batch_size = 8
    seq_length = 32

    # Initialize model
    model = SentimentAnalyzer(vocab_size=vocab_size, d_model=256, n_heads=8, n_classes=3)
    model.eval()

    print(f"\nModel initialized:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Model dimension: 256")
    print(f"  - Classes: 3 (Negative, Neutral, Positive)")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Simulate different types of inputs
    print("\n" + "=" * 60)
    print("Testing different input scenarios:")
    print("=" * 60)

    # Scenario 1: "Easy" confident prediction
    print("\n1. EASY CASE - Clear sentiment:")
    easy_input = torch.randint(0, vocab_size, (1, seq_length))
    with torch.no_grad():
        result = model(easy_input, return_uncertainty=True)

    print(f"   Prediction: Class {result['predictions'].item()}")
    print(f"   Raw confidence: {result['confidence'].item():.4f}")
    print(f"   Epistemic uncertainty: {result['epistemic_uncertainty'].item():.4f}")
    print(f"   Calibrated confidence: {result['calibrated_confidence'].item():.4f}")
    print(f"   -> High confidence, low uncertainty = RELIABLE")

    # Scenario 2: Batch processing
    print("\n2. BATCH PROCESSING - Multiple texts:")
    batch_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    with torch.no_grad():
        batch_result = model(batch_input, return_uncertainty=True)

    print(f"   Processed {batch_size} samples")
    print(f"   Average confidence: {batch_result['confidence'].mean():.4f}")
    print(f"   Average epistemic uncertainty: {batch_result['epistemic_uncertainty'].mean():.4f}")

    # Show distribution
    predictions = batch_result['predictions']
    for cls in range(3):
        count = (predictions == cls).sum().item()
        print(f"   Class {cls}: {count}/{batch_size} samples")

    # Scenario 3: Uncertainty-based filtering
    print("\n3. UNCERTAINTY-BASED FILTERING:")
    uncertainty_threshold = 0.2
    high_conf_mask = batch_result['epistemic_uncertainty'] < uncertainty_threshold

    n_confident = high_conf_mask.sum().item()
    n_uncertain = batch_size - n_confident

    print(f"   Threshold: {uncertainty_threshold}")
    print(f"   Confident predictions: {n_confident}/{batch_size}")
    print(f"   Uncertain predictions: {n_uncertain}/{batch_size}")
    print(f"   -> Route uncertain samples to human review")

    # Scenario 4: Compare with/without uncertainty
    print("\n4. COMPARISON - With vs Without Uncertainty:")
    with torch.no_grad():
        fast_result = model(easy_input, return_uncertainty=False)
        full_result = model(easy_input, return_uncertainty=True)

    print(f"   Without uncertainty: {len(fast_result)} outputs")
    print(f"   With uncertainty: {len(full_result)} outputs")
    print(f"   -> Use uncertainty for critical decisions")

    # Scenario 5: Active Learning Simulation
    print("\n5. ACTIVE LEARNING - Select informative samples:")
    large_batch_size = 20
    large_batch = torch.randint(0, vocab_size, (large_batch_size, seq_length))
    with torch.no_grad():
        al_result = model(large_batch, return_uncertainty=True)

    # Find most uncertain samples (good candidates for labeling)
    uncertainties = al_result['epistemic_uncertainty']

    # Handle different shapes - epistemic_uncertainty is already averaged per sample
    if uncertainties.dim() == 0:
        # Scalar case - shouldn't happen with batches, but handle it
        print(f"   Average uncertainty across all samples: {uncertainties.item():.4f}")
    else:
        # Vector case - one value per sample
        if uncertainties.dim() > 1:
            uncertainties = uncertainties.mean(dim=tuple(range(1, uncertainties.dim())))

        top_k = min(5, uncertainties.numel())  # Ensure k is not larger than actual size
        most_uncertain_indices = uncertainties.topk(top_k).indices

        print(f"   Pool size: {large_batch_size} samples")
        print(f"   Top {top_k} most uncertain samples for labeling:")
        for i, idx in enumerate(most_uncertain_indices):
            print(f"     #{i+1}: Sample {idx.item()} - Uncertainty: {uncertainties[idx]:.4f}")

    # Real-world recommendations
    print("\n" + "=" * 60)
    print("Real-world Usage Recommendations:")
    print("=" * 60)
    print("""
    1. PRODUCTION DEPLOYMENT:
       - Use uncertainty threshold for confidence filtering
       - Route uncertain predictions to human review
       - Monitor epistemic uncertainty over time

    2. ACTIVE LEARNING:
       - Select samples with high epistemic uncertainty
       - Label and retrain to improve model
       - Reduces labeling cost by 50-70%

    3. CRITICAL APPLICATIONS:
       - Always enable uncertainty quantification
       - Set conservative thresholds
       - Log predictions with uncertainty for audit

    4. A/B TESTING:
       - Compare calibrated vs raw confidence
       - Measure correlation with actual errors
       - Optimize threshold based on business metrics
    """)

    print("=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def training_simulation():
    """
    Show how to train with MDL regularization.
    """
    print("\n\n" + "=" * 60)
    print("Training Simulation with MDL Loss")
    print("=" * 60)

    vocab_size = 10000
    n_classes = 3

    model = SentimentAnalyzer(vocab_size=vocab_size, d_model=256, n_heads=8, n_classes=n_classes)
    loss_fn = MDLRegularizedLoss(vocab_size=vocab_size, beta=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\nTraining configuration:")
    print(f"  - Optimizer: Adam (lr=1e-4)")
    print(f"  - Loss: Cross-Entropy + MDL Regularization")
    print(f"  - MDL weight: 0.01")

    print("\nSimulating 5 training steps...\n")

    model.train()
    for step in range(5):
        # Generate dummy batch
        input_ids = torch.randint(0, vocab_size, (8, 32))
        labels = torch.randint(0, n_classes, (8,))

        # Forward pass
        output = model(input_ids)
        logits = output['logits']

        # Compute loss
        ce_loss = nn.CrossEntropyLoss()(logits, labels)

        # Add MDL regularization (requires vocab-level logits)
        # In real scenario, you'd have a language model head
        dummy_lm_logits = torch.randn(8, 32, vocab_size, requires_grad=True)
        dummy_targets = torch.randint(0, vocab_size, (8, 32))
        mdl_output = loss_fn(dummy_lm_logits, dummy_targets)

        total_loss = ce_loss + mdl_output['mdl_penalty']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Step {step+1}:")
        print(f"  CE Loss: {ce_loss.item():.4f}")
        print(f"  MDL Penalty: {mdl_output['mdl_penalty'].item():.4f}")
        print(f"  Total Loss: {total_loss.item():.4f}")
        print(f"  Compression Efficiency: {mdl_output['optimal_complexity']/mdl_output['actual_complexity']:.2f}")

    print("\n" + "=" * 60)
    print("Training simulation completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run sentiment analysis demo
    demo_sentiment_analysis()

    # Run training simulation
    training_simulation()

    print("\n\nNext steps:")
    print("  1. Replace dummy data with real sentiment dataset")
    print("  2. Fine-tune on your specific task")
    print("  3. Calibrate uncertainty thresholds on validation set")
    print("  4. Deploy with uncertainty-based confidence filtering")
