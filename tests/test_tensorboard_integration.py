"""
Test TensorBoard integration with BayesianTransformerLogger
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from src.bayesian_transformer import BayesianTransformerLogger


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_logger_initialization():
    """Test logger initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_exp'
        )

        assert logger.log_dir.exists()
        assert logger.global_step == 0
        logger.close()


def test_log_metrics():
    """Test logging scalar metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_metrics'
        )

        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
        }

        logger.log_metrics(metrics, step=0, prefix='train')
        logger.log_metrics(metrics, step=1, prefix='val')

        logger.close()

        # Verify log directory has event files
        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_log_gradients():
    """Test gradient logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_gradients',
            log_gradients_every=1
        )

        model = SimpleModel()

        # Create dummy gradients
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Log gradients
        logger.log_gradients(model, step=0)
        logger.close()

        # Verify log files exist
        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_log_learning_rate():
    """Test learning rate logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_lr'
        )

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        logger.log_learning_rate(optimizer, step=0)
        logger.close()

        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_log_uncertainty_metrics():
    """Test uncertainty metrics logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_uncertainty'
        )

        epistemic = torch.randn(10, 5).abs()
        aleatoric = torch.randn(10, 5).abs()

        logger.log_uncertainty_metrics(epistemic, aleatoric, step=0)
        logger.close()

        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_log_model_histograms():
    """Test model histogram logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_histograms',
            log_histograms_every=1
        )

        model = SimpleModel()

        # Create dummy gradients
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        logger.log_model_histograms(model, step=0)
        logger.close()

        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_log_attention_stats():
    """Test attention statistics logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_attention'
        )

        # Dummy attention weights and entropy
        attention_weights = torch.softmax(torch.randn(4, 8, 10, 10), dim=-1)
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-9)).sum(dim=-1)

        logger.log_attention_stats(attention_weights, attention_entropy, step=0)
        logger.close()

        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_log_text():
    """Test text logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_text'
        )

        logger.log_text('predictions', 'Sample prediction: positive', step=0)
        logger.close()

        event_files = list(logger.log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_context_manager():
    """Test logger as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_context'
        ) as logger:
            logger.log_metrics({'loss': 0.5}, step=0)
            log_dir = logger.log_dir

        # Verify files were created and closed properly
        event_files = list(log_dir.glob('events.out.tfevents.*'))
        assert len(event_files) > 0


def test_increment_step():
    """Test step counter increment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_step'
        )

        assert logger.global_step == 0
        logger.increment_step()
        assert logger.global_step == 1
        logger.increment_step()
        assert logger.global_step == 2

        logger.close()


def test_training_loop_integration():
    """Test logger in a minimal training loop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with BayesianTransformerLogger(
            log_dir=tmpdir,
            experiment_name='test_training',
            log_gradients_every=1,
            log_histograms_every=2
        ) as logger:

            for epoch in range(3):
                for step in range(5):
                    # Forward pass
                    x = torch.randn(4, 10)
                    y = model(x)
                    loss = y.sum()

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Logging
                    logger.log_metrics({'loss': loss.item()}, prefix='train')
                    logger.log_learning_rate(optimizer)
                    logger.log_gradients(model)
                    logger.log_model_histograms(model)
                    logger.increment_step()

                # Validation logging
                logger.log_metrics({'val_loss': 0.3}, step=epoch, prefix='val')

        # Verify comprehensive logging
        event_files = list(Path(tmpdir).glob('**/*events.out.tfevents.*'))
        assert len(event_files) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
