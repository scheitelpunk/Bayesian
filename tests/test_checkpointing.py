"""
Unit tests for CheckpointManager

Tests the 3-tier checkpointing system:
1. Training checkpoints - Rolling retention
2. Milestone checkpoints - Improvement tracking
3. Production checkpoints - Versioned releases
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from src.bayesian_transformer import CheckpointManager, resume_training


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create optimizer for testing."""
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create checkpoint manager with temp directory."""
    return CheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        max_training_checkpoints=3
    )


class TestCheckpointManagerInit:
    """Test CheckpointManager initialization."""

    def test_directory_creation(self, temp_checkpoint_dir):
        """Test that checkpoint directories are created."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.training_dir.exists()
        assert manager.milestone_dir.exists()
        assert manager.production_dir.exists()
        assert manager.metadata_dir.exists()

    def test_default_parameters(self, temp_checkpoint_dir):
        """Test default parameter values."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.max_training_checkpoints == 3
        assert manager.save_optimizer is True
        assert manager.save_scheduler is True
        assert manager.compress is False


class TestTrainingCheckpoints:
    """Test training checkpoint functionality."""

    def test_save_training_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test saving training checkpoint."""
        metrics = {'loss': 0.5, 'accuracy': 0.85}

        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100, metrics=metrics
        )

        assert Path(filepath).exists()
        assert 'epoch_0000_step_00000100' in filepath

    def test_training_checkpoint_contains_required_keys(
        self, checkpoint_manager, model, optimizer
    ):
        """Test that training checkpoint contains all required keys."""
        metrics = {'loss': 0.5}

        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100, metrics=metrics
        )

        checkpoint = torch.load(filepath, weights_only=False)

        required_keys = [
            'model_state_dict', 'optimizer_state_dict', 'epoch',
            'global_step', 'metrics', 'version', 'checkpoint_type'
        ]

        for key in required_keys:
            assert key in checkpoint

    def test_rolling_retention(self, checkpoint_manager, model, optimizer):
        """Test that only last N training checkpoints are kept."""
        # Save 5 checkpoints (max is 3)
        for i in range(5):
            checkpoint_manager.save_training_checkpoint(
                model, optimizer, epoch=0, step=i*100, metrics={'loss': 0.5}
            )

        checkpoints = list(checkpoint_manager.training_dir.glob('checkpoint_*.pt'))
        assert len(checkpoints) == 3  # Should only keep last 3

    def test_latest_symlink_updated(self, checkpoint_manager, model, optimizer):
        """Test that 'latest' symlink is updated."""
        checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100, metrics={'loss': 0.5}
        )

        latest_link = checkpoint_manager.training_dir / 'latest.pt'
        assert latest_link.exists()


class TestMilestoneCheckpoints:
    """Test milestone checkpoint functionality."""

    def test_save_milestone_on_improvement(self, checkpoint_manager, model):
        """Test that milestone is saved when metric improves."""
        # First checkpoint
        path1 = checkpoint_manager.save_milestone_checkpoint(
            model, metric_value=0.85, metric_name='accuracy',
            epoch=0, config={'d_model': 128}
        )
        assert path1 is not None

        # Worse metric - should not save
        path2 = checkpoint_manager.save_milestone_checkpoint(
            model, metric_value=0.80, metric_name='accuracy',
            epoch=1, config={'d_model': 128}
        )
        assert path2 is None

        # Better metric - should save
        path3 = checkpoint_manager.save_milestone_checkpoint(
            model, metric_value=0.90, metric_name='accuracy',
            epoch=2, config={'d_model': 128}
        )
        assert path3 is not None

    def test_milestone_checkpoint_structure(self, checkpoint_manager, model):
        """Test milestone checkpoint file structure."""
        filepath = checkpoint_manager.save_milestone_checkpoint(
            model, metric_value=0.85, metric_name='accuracy',
            epoch=5, config={'d_model': 128}
        )

        # Check checkpoint file exists
        assert Path(filepath).exists()
        assert 'best_accuracy_0.8500_epoch_0005' in filepath

        # Check JSON metadata file exists
        json_file = Path(filepath).with_suffix('.json')
        assert json_file.exists()

    def test_milestone_indefinite_retention(self, checkpoint_manager, model):
        """Test that milestone checkpoints are not deleted."""
        # Save multiple milestones
        for i in range(10):
            checkpoint_manager.save_milestone_checkpoint(
                model, metric_value=0.80 + i*0.01, metric_name='accuracy',
                epoch=i, config={'d_model': 128}
            )

        checkpoints = list(checkpoint_manager.milestone_dir.glob('best_accuracy_*.pt'))
        assert len(checkpoints) == 10  # All should be kept (symlinks don't count)


class TestProductionCheckpoints:
    """Test production checkpoint functionality."""

    def test_save_production_checkpoint(self, checkpoint_manager, model):
        """Test saving production checkpoint."""
        config = {'d_model': 128, 'n_heads': 4}
        metrics = {'accuracy': 0.92, 'loss': 0.15}
        model_card = {'description': 'Test model'}

        filepath = checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0', config=config,
            metrics=metrics, model_card=model_card
        )

        assert Path(filepath).exists()
        assert 'bayesian_transformer_v1.0.0' in filepath

    def test_production_checkpoint_minimal(self, checkpoint_manager, model):
        """Test that production checkpoint doesn't include optimizer."""
        filepath = checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0',
            config={'d_model': 128},
            metrics={'accuracy': 0.9}
        )

        checkpoint = torch.load(filepath, weights_only=False)

        # Should have model state
        assert 'model_state_dict' in checkpoint

        # Should NOT have optimizer state (production is inference-only)
        assert 'optimizer_state_dict' not in checkpoint

    def test_production_config_json_saved(self, checkpoint_manager, model):
        """Test that config JSON is saved alongside checkpoint."""
        config = {'d_model': 128, 'n_heads': 4}

        filepath = checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0', config=config,
            metrics={'accuracy': 0.9}
        )

        config_file = Path(filepath).with_suffix('.config.json')
        assert config_file.exists()

    def test_production_latest_symlink(self, checkpoint_manager, model):
        """Test that 'latest' symlink points to newest version."""
        checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0', config={'d_model': 128},
            metrics={'accuracy': 0.9}
        )

        checkpoint_manager.save_production_checkpoint(
            model, version='1.1.0', config={'d_model': 128},
            metrics={'accuracy': 0.92}
        )

        latest_link = checkpoint_manager.production_dir / 'latest.pt'
        assert latest_link.exists()


class TestCheckpointLoading:
    """Test checkpoint loading functionality."""

    def test_load_training_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test loading training checkpoint restores state."""
        # Save checkpoint
        original_state = model.state_dict().copy()
        metrics = {'loss': 0.5, 'accuracy': 0.85}

        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=5, step=1000, metrics=metrics
        )

        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0)

        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        metadata = checkpoint_manager.load_checkpoint(
            filepath, new_model, new_optimizer
        )

        # Verify state restored
        assert metadata['epoch'] == 5
        assert metadata['global_step'] == 1000
        assert metadata['metrics']['accuracy'] == 0.85

    def test_load_without_optimizer(self, checkpoint_manager, model):
        """Test loading checkpoint without optimizer."""
        filepath = checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0', config={'d_model': 128},
            metrics={'accuracy': 0.9}
        )

        new_model = SimpleModel()
        metadata = checkpoint_manager.load_checkpoint(filepath, new_model)

        assert 'version' in metadata
        assert metadata['checkpoint_type'] == 'production'

    def test_load_nonexistent_checkpoint_raises_error(self, checkpoint_manager, model):
        """Test that loading nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint('nonexistent.pt', model)

    def test_weights_actually_loaded(self, checkpoint_manager, model, optimizer):
        """Test that model weights are actually restored."""
        # Save original weights
        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=0, metrics={'loss': 0.5}
        )

        original_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Modify weights
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(99.0)

        # Load checkpoint
        new_model = SimpleModel()
        checkpoint_manager.load_checkpoint(filepath, new_model)

        # Verify weights match original
        for name, param in new_model.named_parameters():
            assert torch.allclose(param, original_weights[name])


class TestCheckpointManagement:
    """Test checkpoint management functionality."""

    def test_get_latest_checkpoint(self, checkpoint_manager, model, optimizer):
        """Test getting latest checkpoint."""
        checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100, metrics={'loss': 0.5}
        )

        checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=200, metrics={'loss': 0.4}
        )

        latest = checkpoint_manager.get_latest_checkpoint('training')
        assert latest is not None
        assert latest.exists()

    def test_list_checkpoints(self, checkpoint_manager, model, optimizer):
        """Test listing all checkpoints."""
        # Create some checkpoints
        checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100, metrics={'loss': 0.5}
        )

        checkpoint_manager.save_milestone_checkpoint(
            model, metric_value=0.85, metric_name='accuracy',
            epoch=0, config={'d_model': 128}
        )

        checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0', config={'d_model': 128},
            metrics={'accuracy': 0.9}
        )

        all_checkpoints = checkpoint_manager.list_checkpoints()

        assert 'training' in all_checkpoints
        assert 'milestone' in all_checkpoints
        assert 'production' in all_checkpoints
        assert len(all_checkpoints['training']) >= 1
        assert len(all_checkpoints['milestone']) >= 1
        assert len(all_checkpoints['production']) >= 1

    def test_list_checkpoints_by_type(self, checkpoint_manager, model):
        """Test listing checkpoints filtered by type."""
        checkpoint_manager.save_production_checkpoint(
            model, version='1.0.0', config={'d_model': 128},
            metrics={'accuracy': 0.9}
        )

        checkpoints = checkpoint_manager.list_checkpoints('production')

        assert 'production' in checkpoints
        assert len(checkpoints['production']) == 1


class TestResumeTraining:
    """Test resume training helper function."""

    def test_resume_training_helper(self, checkpoint_manager, model, optimizer):
        """Test resume_training helper function."""
        # Save checkpoint
        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=3, step=500,
            metrics={'loss': 0.3, 'accuracy': 0.88}
        )

        # Create new instances
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters())

        # Resume training
        state = resume_training(str(filepath), new_model, new_optimizer)

        assert state['epoch'] == 3
        assert state['global_step'] == 500
        assert state['metrics']['accuracy'] == 0.88


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_save_checkpoint_with_config(self, checkpoint_manager, model, optimizer):
        """Test saving checkpoint with custom config."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'custom_param': 'value'
        }

        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100,
            metrics={'loss': 0.5}, config=config
        )

        checkpoint = torch.load(filepath, weights_only=False)
        assert checkpoint['config'] == config

    def test_save_without_optimizer(self, checkpoint_manager, model):
        """Test saving training checkpoint without optimizer."""
        manager = CheckpointManager(
            checkpoint_dir=checkpoint_manager.checkpoint_dir,
            save_optimizer=False
        )

        filepath = manager.save_training_checkpoint(
            model, optimizer=None, epoch=0, step=100,
            metrics={'loss': 0.5}
        )

        checkpoint = torch.load(filepath, weights_only=False)
        assert 'optimizer_state_dict' not in checkpoint

    def test_metadata_includes_environment_info(
        self, checkpoint_manager, model, optimizer
    ):
        """Test that metadata includes environment information."""
        filepath = checkpoint_manager.save_training_checkpoint(
            model, optimizer, epoch=0, step=100, metrics={'loss': 0.5}
        )

        checkpoint = torch.load(filepath, weights_only=False)

        assert 'metadata' in checkpoint
        assert 'pytorch_version' in checkpoint['metadata']
        assert 'python_version' in checkpoint['metadata']
        assert 'hostname' in checkpoint['metadata']


# Integration test
def test_complete_workflow(temp_checkpoint_dir):
    """Test complete checkpoint workflow."""
    # Initialize
    manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simulate training loop
    for epoch in range(3):
        for step in range(100):
            if step % 50 == 0:
                manager.save_training_checkpoint(
                    model, optimizer, epoch, step,
                    metrics={'loss': 0.5 - epoch*0.1}
                )

        # Save milestone
        val_acc = 0.8 + epoch * 0.05
        manager.save_milestone_checkpoint(
            model, val_acc, 'accuracy', epoch,
            config={'d_model': 128}
        )

    # Save production
    manager.save_production_checkpoint(
        model, version='1.0.0', config={'d_model': 128},
        metrics={'accuracy': 0.9}
    )

    # Verify checkpoints exist
    all_checkpoints = manager.list_checkpoints()
    assert len(all_checkpoints['training']) == 3  # Rolled over
    # Milestone count includes actual files, not symlinks
    assert len([c for c in all_checkpoints['milestone'] if c.name.startswith('best_accuracy_')]) == 3
    assert len(all_checkpoints['production']) == 1

    # Load and verify
    latest = manager.get_latest_checkpoint('production')
    new_model = SimpleModel()
    metadata = manager.load_checkpoint(latest, new_model)
    assert metadata['version'] == '1.0.0'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
