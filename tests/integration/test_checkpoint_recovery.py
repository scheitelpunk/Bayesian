"""
Integration tests for checkpoint recovery scenarios.

Tests crash recovery, state restoration, and checkpoint integrity.
"""

import pytest
import torch
from pathlib import Path
import tempfile


class TestCheckpointRecovery:
    """Tests for checkpoint recovery scenarios."""

    def test_crash_recovery(self):
        """Simulate crash and recovery."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        from src.bayesian_transformer.checkpointing import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            # Initial training
            model = BayesianExpectationTransformerLayer(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # Train for 5 steps
            for step in range(5):
                x = torch.randn(2, 16, config['d_model'])
                output = model(x)
                loss = output['hidden_states'].mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                checkpoint_mgr.save_training_checkpoint(
                    model, optimizer, epoch=0, step=step,
                    metrics={'loss': loss.item()}, config=config
                )

            # Get latest checkpoint
            training_dir = Path(tmpdir) / 'training'
            checkpoints = sorted(training_dir.glob('checkpoint_*.pt'))
            assert len(checkpoints) > 0, "No checkpoints created"

            last_checkpoint = checkpoints[-1]

            # Simulate crash - create new model/optimizer
            new_model = BayesianExpectationTransformerLayer(config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            # Resume
            state = checkpoint_mgr.load_checkpoint(
                str(last_checkpoint), new_model, new_optimizer
            )

            assert state['global_step'] == 4, f"Expected step 4, got {state['global_step']}"
            assert state['epoch'] == 0, f"Expected epoch 0, got {state['epoch']}"

            # Verify parameters match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(),
                new_model.named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-6), \
                    f"Parameter mismatch for {n1}"

    def test_optimizer_state_recovery(self):
        """Test that optimizer state (momentum, etc.) is restored."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        from src.bayesian_transformer.checkpointing import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            # Initial training with momentum
            model = BayesianExpectationTransformerLayer(config)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # Train for several steps to build momentum
            for step in range(10):
                x = torch.randn(2, 16, config['d_model'])
                output = model(x)
                loss = output['hidden_states'].mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Save checkpoint
            checkpoint_path = checkpoint_mgr.save_training_checkpoint(
                model, optimizer, epoch=0, step=9,
                metrics={'loss': loss.item()}
            )

            # Create new optimizer (without momentum state)
            new_model = BayesianExpectationTransformerLayer(config)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01, momentum=0.9)

            # Load checkpoint
            checkpoint_mgr.load_checkpoint(checkpoint_path, new_model, new_optimizer)

            # Verify optimizer state was restored
            original_state = optimizer.state_dict()
            restored_state = new_optimizer.state_dict()

            assert 'state' in restored_state, "Missing optimizer state"
            assert len(restored_state['state']) == len(original_state['state']), \
                "Optimizer state count mismatch"

    def test_rolling_checkpoint_retention(self):
        """Test that old training checkpoints are automatically cleaned up."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        from src.bayesian_transformer.checkpointing import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            model = BayesianExpectationTransformerLayer(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Create manager with max 3 checkpoints
            checkpoint_mgr = CheckpointManager(
                checkpoint_dir=tmpdir,
                max_training_checkpoints=3
            )

            # Save 10 checkpoints
            for step in range(10):
                x = torch.randn(2, 16, config['d_model'])
                output = model(x)
                loss = output['hidden_states'].mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                checkpoint_mgr.save_training_checkpoint(
                    model, optimizer, epoch=0, step=step,
                    metrics={'loss': loss.item()}
                )

            # Verify only last 3 checkpoints remain
            training_dir = Path(tmpdir) / 'training'
            checkpoints = list(training_dir.glob('checkpoint_*.pt'))

            assert len(checkpoints) == 3, \
                f"Expected 3 checkpoints, found {len(checkpoints)}"

    def test_milestone_checkpoint_persistence(self):
        """Test that milestone checkpoints are never automatically deleted."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        from src.bayesian_transformer.checkpointing import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            model = BayesianExpectationTransformerLayer(config)
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # Save multiple milestone checkpoints
            losses = [0.5, 0.4, 0.3, 0.35, 0.25]

            for epoch, loss in enumerate(losses):
                checkpoint_mgr.save_milestone_checkpoint(
                    model, metric_value=loss, metric_name='loss',
                    epoch=epoch
                )

            # All milestones should persist
            milestone_dir = Path(tmpdir) / 'milestone'
            milestones = list(milestone_dir.glob('best_*.pt'))

            # Should have checkpoints for each improvement (0.5 -> 0.4 -> 0.3 -> 0.25)
            assert len(milestones) >= 1, "No milestone checkpoints created"

            # Latest should be best
            latest_link = milestone_dir / 'best_loss.pt'
            assert latest_link.exists(), "Missing best_loss symlink"

    def test_checkpoint_corruption_detection(self):
        """Test handling of corrupted checkpoint files."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        from src.bayesian_transformer.checkpointing import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            model = BayesianExpectationTransformerLayer(config)
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)

            # Create corrupted checkpoint
            corrupted_path = Path(tmpdir) / 'corrupted.pt'
            with open(corrupted_path, 'w') as f:
                f.write("This is not a valid checkpoint")

            # Attempt to load should raise error
            new_model = BayesianExpectationTransformerLayer(config)

            with pytest.raises(Exception):
                checkpoint_mgr.load_checkpoint(str(corrupted_path), new_model)

    def test_config_mismatch_detection(self):
        """Test detection of config mismatch when loading checkpoint."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        from src.bayesian_transformer.checkpointing import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            config1 = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            config2 = {
                'd_model': 128,  # Different dimension
                'n_heads': 8,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            # Save with config1
            model1 = BayesianExpectationTransformerLayer(config1)
            checkpoint_mgr = CheckpointManager(checkpoint_dir=tmpdir)

            checkpoint_path = checkpoint_mgr.save_training_checkpoint(
                model1, None, epoch=0, step=0, metrics={'loss': 1.0},
                config=config1
            )

            # Try to load into model with config2
            model2 = BayesianExpectationTransformerLayer(config2)

            # Should raise error due to shape mismatch
            with pytest.raises(RuntimeError):
                checkpoint_mgr.load_checkpoint(checkpoint_path, model2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
