"""
Checkpointing System Demo

Demonstrates the 3-tier checkpointing system:
1. Training checkpoints - Frequent, rolling retention
2. Milestone checkpoints - On improvement, indefinite retention
3. Production checkpoints - Versioned, minimal storage

This demo shows how to integrate checkpointing into training workflows
and how to resume training from saved checkpoints.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.bayesian_transformer import (
    CheckpointManager,
    resume_training
)

# Import from real_data_demo
from real_data_demo import (
    BayesianSentimentClassifier,
    load_imdb_data,
    SimpleTokenizer,
    create_dataloader,
    evaluate
)


def train_epoch_with_checkpointing(
    model, dataloader, optimizer, device,
    checkpoint_mgr, epoch, save_every=100
):
    """Train for one epoch with periodic checkpointing.

    Args:
        model: PyTorch model
        dataloader: Training dataloader
        optimizer: Optimizer instance
        device: Device to train on
        checkpoint_mgr: CheckpointManager instance
        epoch: Current epoch number
        save_every: Save checkpoint every N steps

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    global_step = epoch * len(dataloader)

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(input_ids)
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

        # Save training checkpoint every N steps
        if step % save_every == 0:
            metrics = {
                'loss': loss.item(),
                'accuracy': correct / total if total > 0 else 0,
                'samples_processed': total
            }
            checkpoint_mgr.save_training_checkpoint(
                model, optimizer, epoch, global_step + step, metrics
            )

    return total_loss / len(dataloader), correct / total


def demo_checkpointing_system():
    """Demonstrate complete checkpointing workflow."""

    print("=" * 70)
    print("CHECKPOINTING SYSTEM DEMONSTRATION")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Initialize checkpoint manager
    print("\n1. Initializing Checkpoint Manager...")
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir='checkpoints',
        max_training_checkpoints=3,  # Keep only last 3 training checkpoints
        save_optimizer=True,
        save_scheduler=False,
        compress=False  # Can enable for production
    )

    print(f"   Checkpoint directory: {checkpoint_mgr.checkpoint_dir}")
    print(f"   Max training checkpoints: {checkpoint_mgr.max_training_checkpoints}")

    # Load data (smaller dataset for quick demo)
    print("\n2. Loading IMDB dataset...")
    train_dataset = load_imdb_data(split='train', streaming=True, max_samples=500)
    test_dataset = load_imdb_data(split='test', streaming=True, max_samples=100)

    # Create tokenizer
    print("\n3. Building vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=5000, max_length=64)
    train_texts = [item['text'] for item in train_dataset]
    tokenizer.build_vocab(train_texts)

    # Create dataloaders
    train_loader = create_dataloader(train_dataset, tokenizer, batch_size=16, max_length=64)
    test_loader = create_dataloader(test_dataset, tokenizer, batch_size=16, max_length=64)

    # Initialize model
    print("\n4. Initializing model...")
    vocab_size = len(tokenizer.vocab)
    model = BayesianSentimentClassifier(vocab_size=vocab_size, d_model=64, n_heads=2)
    model = model.to(device)

    # Model configuration for checkpoints
    model_config = {
        'd_model': 64,
        'n_heads': 2,
        'vocab_size': vocab_size,
        'max_length': 64,
        'architecture': 'BayesianExpectationTransformer'
    }

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 3
    best_val_acc = 0.0

    print(f"\n5. Training with checkpointing for {n_epochs} epochs...")
    print("-" * 70)

    for epoch in range(n_epochs):
        # Train with checkpointing
        train_loss, train_acc = train_epoch_with_checkpointing(
            model, train_loader, optimizer, device,
            checkpoint_mgr=checkpoint_mgr,
            epoch=epoch,
            save_every=50  # Save every 50 steps
        )

        print(f"   Epoch {epoch+1}/{n_epochs}:")
        print(f"     Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validate
        val_results = evaluate(model, test_loader, device, use_uncertainty=False)
        val_acc = val_results['accuracy']
        print(f"     Val Accuracy: {val_acc:.4f}")

        # Save milestone checkpoint if validation improved
        if val_acc > best_val_acc:
            metrics = {
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc
            }

            checkpoint_mgr.save_milestone_checkpoint(
                model,
                metric_value=val_acc,
                metric_name='accuracy',
                epoch=epoch,
                config=model_config,
                metadata={
                    'notes': f'Best validation accuracy at epoch {epoch+1}',
                    'train_samples': len(train_dataset),
                    'val_samples': len(test_dataset)
                },
                optimizer=optimizer
            )
            best_val_acc = val_acc

    print("-" * 70)

    # Save final production checkpoint
    print("\n6. Saving production checkpoint...")
    final_metrics = {
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'val_accuracy': best_val_acc,
        'final_epoch': n_epochs
    }

    model_card = {
        'description': 'Bayesian Expectation Transformer for sentiment analysis',
        'dataset': 'IMDB movie reviews',
        'intended_use': 'Binary sentiment classification with uncertainty quantification',
        'training_epochs': n_epochs,
        'training_samples': len(train_dataset),
        'performance': final_metrics,
        'limitations': 'Trained on limited dataset for demonstration purposes',
        'ethical_considerations': 'Model may reflect biases present in IMDB reviews'
    }

    production_path = checkpoint_mgr.save_production_checkpoint(
        model,
        version='1.0.0',
        config=model_config,
        metrics=final_metrics,
        model_card=model_card
    )

    # Demonstrate checkpoint listing
    print("\n7. Listing all checkpoints...")
    all_checkpoints = checkpoint_mgr.list_checkpoints()

    print(f"\n   Training checkpoints ({len(all_checkpoints['training'])}):")
    for cp in all_checkpoints['training']:
        print(f"     - {cp.name}")

    print(f"\n   Milestone checkpoints ({len(all_checkpoints['milestone'])}):")
    for cp in all_checkpoints['milestone']:
        print(f"     - {cp.name}")

    print(f"\n   Production checkpoints ({len(all_checkpoints['production'])}):")
    for cp in all_checkpoints['production']:
        print(f"     - {cp.name}")

    # Demonstrate loading checkpoint
    print("\n8. Loading production checkpoint...")
    latest_prod = checkpoint_mgr.get_latest_checkpoint('production')

    if latest_prod:
        # Create new model instance
        new_model = BayesianSentimentClassifier(vocab_size=vocab_size, d_model=64, n_heads=2)
        new_model = new_model.to(device)

        # Load checkpoint
        metadata = checkpoint_mgr.load_checkpoint(
            latest_prod, new_model, device=device
        )

        print(f"\n   Loaded checkpoint metadata:")
        print(f"     Version: {metadata['version']}")
        print(f"     Type: {metadata['checkpoint_type']}")
        print(f"     Config: {metadata['config']}")
        print(f"     Metrics: {metadata['metrics']}")

        # Verify loaded model works
        print("\n9. Verifying loaded model...")
        val_results = evaluate(new_model, test_loader, device, use_uncertainty=False)
        print(f"   Loaded model accuracy: {val_results['accuracy']:.4f}")
        print(f"   Original model accuracy: {best_val_acc:.4f}")
        print(f"   Match: {'✓' if abs(val_results['accuracy'] - best_val_acc) < 0.01 else '✗'}")

    # Demonstrate resume training
    print("\n10. Demonstrating resume training...")
    latest_training = checkpoint_mgr.get_latest_checkpoint('training')

    if latest_training:
        print(f"\n   Latest training checkpoint: {latest_training.name}")

        # Create new instances
        resume_model = BayesianSentimentClassifier(vocab_size=vocab_size, d_model=64, n_heads=2)
        resume_model = resume_model.to(device)
        resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=1e-3)

        # Resume training
        state = resume_training(
            str(latest_training),
            resume_model,
            resume_optimizer,
            device=device
        )

        start_epoch = state['epoch'] + 1
        print(f"\n   Would resume training from epoch {start_epoch}")
        print(f"   Previous step: {state['global_step']}")
        print(f"   Previous metrics: {state['metrics']}")

    # Summary
    print("\n" + "=" * 70)
    print("CHECKPOINTING SYSTEM FEATURES DEMONSTRATED")
    print("=" * 70)
    print("""
✓ Training Checkpoints:
  - Saved periodically during training (every N steps)
  - Rolling retention (keeps last 3)
  - Includes full state for exact resume

✓ Milestone Checkpoints:
  - Saved when validation metric improves
  - Indefinite retention
  - Includes metadata and notes

✓ Production Checkpoints:
  - Versioned (semantic versioning)
  - Minimal size (no optimizer state)
  - Includes comprehensive model card

✓ Resume Training:
  - Load checkpoint with full state
  - Continue from exact point
  - Preserves optimizer momentum

✓ Checkpoint Management:
  - List all checkpoints by type
  - Get latest checkpoint
  - Atomic writes (no corruption)
  - Cross-platform compatible
    """)

    print("\nCheckpoint locations:")
    print(f"  Training:   {checkpoint_mgr.training_dir}")
    print(f"  Milestone:  {checkpoint_mgr.milestone_dir}")
    print(f"  Production: {checkpoint_mgr.production_dir}")
    print(f"  Metadata:   {checkpoint_mgr.metadata_dir}")

    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("""
# Initialize checkpoint manager
from src.bayesian_transformer import CheckpointManager

manager = CheckpointManager(checkpoint_dir='checkpoints')

# During training loop
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        # ... training code ...

        if step % 100 == 0:
            manager.save_training_checkpoint(
                model, optimizer, epoch, step, metrics
            )

    # After validation
    if val_acc > best_acc:
        manager.save_milestone_checkpoint(
            model, val_acc, 'accuracy', epoch, config
        )

# Save production version
manager.save_production_checkpoint(
    model, version='1.0.0', config=config,
    metrics=metrics, model_card=model_card
)

# Resume training
from src.bayesian_transformer import resume_training

state = resume_training('checkpoints/training/latest.pt',
                       model, optimizer)
start_epoch = state['epoch'] + 1
    """)

    print("=" * 70)


if __name__ == "__main__":
    demo_checkpointing_system()

    print("\n\nNEXT STEPS:")
    print("  1. Integrate checkpointing into your training script")
    print("  2. Configure checkpoint retention policies")
    print("  3. Set up cloud storage sync for backups")
    print("  4. Implement automated checkpoint cleanup")
    print("  5. Add checkpoint validation tests")
