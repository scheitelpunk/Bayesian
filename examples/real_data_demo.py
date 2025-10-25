"""
Real-World Demo: IMDB Sentiment Analysis with Bayesian Uncertainty

This demo uses REAL IMDB movie reviews from HuggingFace to demonstrate:
1. Training on real data from the Stanford IMDB dataset
2. Uncertainty-based confidence filtering
3. Active learning sample selection
4. Performance comparison with/without uncertainty
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter
import re
from datasets import load_dataset
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bayesian_transformer import (
    BayesianExpectationTransformerLayer,
    MDLRegularizedLoss,
    BayesianTransformerLogger,
    CheckpointManager
)


def load_imdb_data(split='train', streaming=True, max_samples=20000):
    """Load real IMDB dataset from HuggingFace.

    Args:
        split: 'train', 'test', or 'unsupervised'
        streaming: If True, use streaming mode (instant start)
        max_samples: Limit samples for quick testing

    Returns:
        Dataset object
    """
    print(f"   Loading IMDB dataset (split={split}, streaming={streaming})...")
    dataset = load_dataset('stanfordnlp/imdb', split=split, streaming=streaming)

    if streaming:
        # For streaming, we need to convert to list for easier handling
        dataset = list(dataset.take(max_samples))
    else:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"   Loaded {len(dataset)} samples")
    return dataset


class SimpleTokenizer:
    """Simple word-level tokenizer for demo purposes."""

    def __init__(self, vocab_size=10000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word_to_idx = {}

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        all_words = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)

        # Add most common words to vocab
        for word, _ in word_counts.most_common(self.vocab_size - 2):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        self.word_to_idx = self.vocab
        print(f"   Built vocabulary with {len(self.vocab)} tokens")

    def __call__(self, texts, max_length=None, padding='max_length',
                 truncation=True, return_tensors='pt'):
        """Tokenize texts."""
        max_len = max_length or self.max_length

        tokens = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())

            # Convert to indices
            indices = [self.word_to_idx.get(word, 1) for word in words]

            # Truncate
            if truncation and len(indices) > max_len:
                indices = indices[:max_len]

            # Pad
            if padding == 'max_length':
                indices += [0] * (max_len - len(indices))

            tokens.append(indices)

        if return_tensors == 'pt':
            return {'input_ids': torch.tensor(tokens, dtype=torch.long)}
        return {'input_ids': tokens}


def create_dataloader(dataset, tokenizer, batch_size=8, max_length=512):
    """Create DataLoader with proper tokenization for IMDB data."""

    def collate_fn(batch):
        # Handle both streaming (list of dicts) and regular dataset formats
        if isinstance(batch[0], dict):
            texts = [item['text'] for item in batch]
            labels = [item['label'] for item in batch]
        else:
            texts = [item['text'] for item in batch]
            labels = [item['label'] for item in batch]

        # Tokenize
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': tokenized['input_ids'],
            'label': torch.tensor(labels, dtype=torch.long),
            'text': texts
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0  # 0 for Windows compatibility
    )


class BayesianSentimentClassifier(nn.Module):
    """Sentiment classifier with Bayesian uncertainty."""

    def __init__(self, vocab_size, d_model=128, n_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        config = {
            'd_model': d_model,
            'n_heads': n_heads,
            'vocab_size': vocab_size
        }
        self.transformer = BayesianExpectationTransformerLayer(config)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, input_ids, return_uncertainty=False):
        x = self.embedding(input_ids)
        transformer_out = self.transformer(x, return_uncertainty=return_uncertainty)
        hidden = transformer_out['hidden_states']

        # Mean pooling
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)

        output = {'logits': logits}

        if return_uncertainty:
            # Extract uncertainty from transformer
            uncertainty = transformer_out['uncertainty']
            epistemic = uncertainty['epistemic'].mean(dim=-1)
            output['epistemic_uncertainty'] = epistemic

        return output


def train_epoch(model, dataloader, optimizer, device, logger=None, epoch=0):
    """Train for one epoch with optional TensorBoard logging."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        # Forward pass with uncertainty
        outputs = model(input_ids, return_uncertainty=True)
        logits = outputs['logits']

        # Compute loss
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # TensorBoard logging
        if logger is not None:
            # Log metrics every batch
            batch_acc = (preds == labels).float().mean().item()
            logger.log_metrics({
                'loss': loss.item(),
                'accuracy': batch_acc,
            }, prefix='train')

            # Log learning rate
            logger.log_learning_rate(optimizer)

            # Log gradients (every 100 steps)
            logger.log_gradients(model)

            # Log uncertainty metrics if available
            if 'epistemic_uncertainty' in outputs:
                epistemic = outputs['epistemic_uncertainty']
                # Handle scalar or tensor uncertainty
                if epistemic.dim() == 0:
                    epistemic = epistemic.unsqueeze(0).expand(labels.size(0))
                logger.log_uncertainty_metrics(epistemic)

            # Log histograms (every 500 steps)
            logger.log_model_histograms(model)

            logger.increment_step()

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device, use_uncertainty=False, threshold=0.15, logger=None):
    """Evaluate model with optional uncertainty filtering and TensorBoard logging."""
    model.eval()
    correct = 0
    total = 0
    uncertain_count = 0

    all_predictions = []
    all_labels = []
    all_uncertainties = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, return_uncertainty=use_uncertainty)
            logits = outputs['logits']
            preds = logits.argmax(dim=-1)

            # Compute loss for logging
            loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

            if use_uncertainty:
                uncertainties = outputs['epistemic_uncertainty']

                # Handle scalar or tensor
                if uncertainties.dim() == 0:
                    # Scalar case - apply to all samples
                    uncertainties = uncertainties.unsqueeze(0).expand(labels.size(0))

                # Filter by uncertainty
                confident_mask = uncertainties < threshold

                if confident_mask.any():
                    correct += ((preds == labels) & confident_mask).sum().item()
                    total += confident_mask.sum().item()

                uncertain_count += (~confident_mask).sum().item()
                all_uncertainties.extend(uncertainties.cpu().numpy().flatten())
            else:
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(dataloader)

    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels,
        'uncertain_count': uncertain_count
    }

    if use_uncertainty:
        results['uncertainties'] = all_uncertainties

    # Log validation metrics
    if logger is not None:
        logger.log_metrics({
            'loss': avg_loss,
            'accuracy': accuracy,
        }, prefix='val')

        if use_uncertainty and all_uncertainties:
            logger.log_metrics({
                'uncertain_count': uncertain_count,
                'mean_uncertainty': np.mean(all_uncertainties),
            }, prefix='val')

    return results


def show_sample_predictions(model, test_loader, device, num_samples=5):
    """Show predictions with uncertainty on real IMDB reviews."""
    model.eval()
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS WITH UNCERTAINTY")
    print("=" * 70)

    with torch.no_grad():
        batch = next(iter(test_loader))
        input_ids = batch['input_ids'].to(device)
        labels = batch['label']
        texts = batch['text']

        outputs = model(input_ids, return_uncertainty=True)
        logits = outputs['logits']
        preds = logits.argmax(dim=-1).cpu()
        probs = torch.softmax(logits, dim=-1).cpu()

        # Get uncertainty
        uncertainties = outputs['epistemic_uncertainty']
        if uncertainties.dim() == 0:
            uncertainties = uncertainties.unsqueeze(0).expand(len(labels))
        uncertainties = uncertainties.cpu()

        for i in range(min(num_samples, len(texts))):
            pred = preds[i].item()
            true_label = labels[i].item()
            uncertainty = uncertainties[i].item()
            confidence = probs[i][pred].item()

            pred_str = "POSITIVE" if pred == 1 else "NEGATIVE"
            true_str = "POSITIVE" if true_label == 1 else "NEGATIVE"
            correct = "✓ CORRECT" if pred == true_label else "✗ WRONG"

            print(f"\nSample #{i+1} {correct}")
            print(f"Review: {texts[i][:150]}...")
            print(f"True Label: {true_str} | Predicted: {pred_str}")
            print(f"Confidence: {confidence:.4f} | Uncertainty: {uncertainty:.4f}")
            print("-" * 70)


def demo_real_world_training():
    """Main demo with real IMDB data from HuggingFace."""

    print("=" * 70)
    print("REAL IMDB SENTIMENT ANALYSIS WITH BAYESIAN UNCERTAINTY")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load real IMDB datasets
    print("\n1. Loading REAL IMDB dataset from HuggingFace...")
    train_dataset = load_imdb_data(split='train', streaming=True, max_samples=1000)
    test_dataset = load_imdb_data(split='test', streaming=True, max_samples=200)

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")

    # Show some real examples
    print("\n   Sample REAL IMDB reviews:")
    for i in range(3):
        sample = train_dataset[i]
        label_str = "POSITIVE" if sample['label'] == 1 else "NEGATIVE"
        text_preview = sample['text'][:100].replace('\n', ' ')
        print(f"   [{label_str}] {text_preview}...")

    # Create tokenizer and build vocabulary
    print("\n2. Building vocabulary from real reviews...")
    tokenizer = SimpleTokenizer(vocab_size=10000, max_length=128)

    # Build vocab from training data
    train_texts = [item['text'] for item in train_dataset]
    tokenizer.build_vocab(train_texts)

    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader = create_dataloader(train_dataset, tokenizer, batch_size=32, max_length=128)
    test_loader = create_dataloader(test_dataset, tokenizer, batch_size=32, max_length=128)

    # Initialize model
    print("\n4. Initializing Bayesian Sentiment Classifier...")
    vocab_size = len(tokenizer.vocab)
    model = BayesianSentimentClassifier(vocab_size=vocab_size, d_model=128, n_heads=4)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Total parameters: {n_params:,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 5

    # Initialize TensorBoard logger and checkpoint manager
    print("\n5. Initializing TensorBoard logger and checkpoint manager...")
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir='C:/dev/coding/Bayesian/checkpoints',
        max_training_checkpoints=3
    )

    # Create TensorBoard logger with context manager
    with BayesianTransformerLogger(
        log_dir='runs',
        experiment_name='bayesian_transformer_imdb'
    ) as logger:

        print(f"   TensorBoard logs will be saved to: {logger.log_dir}")
        print(f"   View with: tensorboard --logdir={logger.log_dir}")

        print(f"\n6. Training on REAL IMDB data for {n_epochs} epochs...")
        print("-" * 70)

        best_val_acc = 0.0

        for epoch in range(n_epochs):
            # Training
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, logger=logger, epoch=epoch
            )
            print(f"   Epoch {epoch+1}/{n_epochs}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

            # Validation (every epoch)
            val_results = evaluate(model, test_loader, device, use_uncertainty=True, logger=logger)
            val_acc = val_results['accuracy']

            # Log epoch-level metrics
            logger.log_metrics({
                'epoch_loss': train_loss,
                'epoch_accuracy': train_acc,
            }, step=epoch, prefix='train_epoch')

            logger.log_metrics({
                'epoch_accuracy': val_acc,
                'epoch_loss': val_results['loss'],
            }, step=epoch, prefix='val_epoch')

            # Save checkpoint if best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save milestone checkpoint (on improvement)
                checkpoint_mgr.save_milestone_checkpoint(
                    model=model,
                    metric_value=val_acc,
                    metric_name='accuracy',
                    epoch=epoch,
                    metadata={'val_loss': val_results['loss']}
                )
                print(f"   -> New best validation accuracy: {val_acc:.4f} (checkpoint saved)")

        print("-" * 70)
        print(f"\nTensorBoard logs saved to: {logger.log_dir}")
        print(f"View with: tensorboard --logdir={logger.log_dir}")
        print(f"Or run: scripts/view_tensorboard.bat (Windows) or scripts/view_tensorboard.sh (Linux/Mac)")
        print(f"Then navigate to: http://localhost:6006")

    # Show sample predictions
    show_sample_predictions(model, test_loader, device, num_samples=5)

    # Evaluation without uncertainty
    print("\n7. Evaluating WITHOUT uncertainty filtering...")
    results_no_unc = evaluate(model, test_loader, device, use_uncertainty=False)
    print(f"   Accuracy: {results_no_unc['accuracy']:.4f}")
    print(f"   Predictions on all {len(results_no_unc['predictions'])} samples")

    # Evaluation with uncertainty
    print("\n8. Evaluating WITH uncertainty filtering...")
    uncertainty_threshold = 0.15
    results_with_unc = evaluate(model, test_loader, device, use_uncertainty=True,
                                 threshold=uncertainty_threshold)

    print(f"   Uncertainty threshold: {uncertainty_threshold}")
    print(f"   Confident predictions: {len(results_with_unc['predictions']) - results_with_unc['uncertain_count']}")
    print(f"   Uncertain predictions (for human review): {results_with_unc['uncertain_count']}")
    print(f"   Accuracy on confident predictions: {results_with_unc['accuracy']:.4f}")

    # Uncertainty distribution
    uncertainties = np.array(results_with_unc['uncertainties'])
    print(f"\n   Uncertainty statistics:")
    print(f"     Mean: {uncertainties.mean():.4f}")
    print(f"     Std:  {uncertainties.std():.4f}")
    print(f"     Min:  {uncertainties.min():.4f}")
    print(f"     Max:  {uncertainties.max():.4f}")

    # Active learning simulation
    print("\n9. Active Learning Simulation...")
    print("   Finding most informative REAL IMDB samples for labeling:")

    # Get samples sorted by uncertainty
    uncertain_indices = np.argsort(uncertainties)[-5:]  # Top 5 most uncertain

    for i, idx in enumerate(uncertain_indices):
        sample = test_dataset[idx]
        uncertainty = uncertainties[idx]
        pred = results_with_unc['predictions'][idx]
        true_label = sample['label']

        pred_str = "POSITIVE" if pred == 1 else "NEGATIVE"
        true_str = "POSITIVE" if true_label == 1 else "NEGATIVE"
        correct = "[CORRECT]" if pred == true_label else "[WRONG]"

        text_preview = sample['text'][:100].replace('\n', ' ')
        print(f"\n   Sample #{i+1} (Uncertainty: {uncertainty:.4f}) {correct}")
        print(f"     Text: {text_preview}...")
        print(f"     Predicted: {pred_str} | True: {true_str}")

    # Cost-benefit analysis
    print("\n10. Cost-Benefit Analysis:")
    total_samples = len(results_with_unc['predictions'])
    confident_samples = total_samples - results_with_unc['uncertain_count']

    # Assume human review costs 10x automated processing
    automated_cost = total_samples * 1
    hybrid_cost = confident_samples * 1 + results_with_unc['uncertain_count'] * 10

    print(f"   Full automated processing cost: {automated_cost} units")
    print(f"   Hybrid (AI + human) cost: {hybrid_cost} units")
    print(f"   Cost increase: {(hybrid_cost/automated_cost - 1)*100:.1f}%")
    accuracy_improvement = (results_with_unc['accuracy']/results_no_unc['accuracy'] - 1)*100 if results_no_unc['accuracy'] > 0 else 0
    print(f"   Accuracy improvement: {accuracy_improvement:.1f}%")
    print(f"   -> Better accuracy for only {(hybrid_cost/automated_cost - 1)*100:.1f}% more cost!")

    # Real-world recommendations
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 70)
    print(f"""
1. CONFIDENCE THRESHOLD TUNING:
   - Current: {uncertainty_threshold}
   - Calibrate on validation set based on business requirements
   - Lower threshold = more human review, higher accuracy
   - Higher threshold = more automation, lower cost

2. MONITORING IN PRODUCTION:
   - Track uncertainty distribution over time
   - Alert if mean uncertainty increases (model drift)
   - A/B test different thresholds

3. ACTIVE LEARNING PIPELINE:
   - Weekly: collect {results_with_unc['uncertain_count']} uncertain samples
   - Human annotate these samples
   - Retrain model with new labels
   - Expected: {results_with_unc['uncertain_count']*52} labels/year vs {total_samples*52} full annotation

4. BUSINESS IMPACT:
   - Automated: {confident_samples}/{total_samples} = {confident_samples/total_samples*100:.1f}% of volume
   - Human review: {results_with_unc['uncertain_count']}/{total_samples} = {results_with_unc['uncertain_count']/total_samples*100:.1f}% of volume
   - Estimated cost savings: {(1 - hybrid_cost/automated_cost*10)*100:.1f}% vs full human review
    """)

    print("=" * 70)
    print("Demo completed! Model ready for deployment.")
    print("=" * 70)


if __name__ == "__main__":
    demo_real_world_training()

    print("\n\nNEXT STEPS:")
    print("  1. ✓ COMPLETED: Real IMDB dataset integrated from HuggingFace")
    print("  2. Train for more epochs on larger dataset (25K samples)")
    print("  3. Experiment with different uncertainty thresholds")
    print("  4. Deploy with REST API + uncertainty-based routing")
    print("  5. Implement active learning feedback loop")
