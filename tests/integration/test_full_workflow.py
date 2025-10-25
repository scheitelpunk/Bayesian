"""
Integration tests for complete training workflow.

Tests the full pipeline including model training, checkpointing,
and logging to ensure all components work together correctly.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
from src.bayesian_transformer import BayesianExpectationTransformerLayer


class TestFullWorkflow:
    """Integration tests for complete training workflow."""

    def test_training_with_checkpointing(self):
        """Test complete workflow: train, checkpoint, resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize model and optimizer
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,  # Reduced for faster testing
                'dropout': 0.1
            }

            model = BayesianExpectationTransformerLayer(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Import checkpoint manager
            from src.bayesian_transformer.checkpointing import CheckpointManager
            checkpoint_mgr = CheckpointManager(checkpoint_dir=f'{tmpdir}/checkpoints')

            # Simulate training for 2 epochs
            best_loss = float('inf')
            for epoch in range(2):
                epoch_losses = []

                for step in range(10):
                    # Generate random batch
                    x = torch.randn(2, 16, config['d_model'])

                    # Forward pass
                    output = model(x, return_uncertainty=True, generate_cot=True)

                    # Compute simple loss
                    loss = output['hidden_states'].mean()
                    epoch_losses.append(loss.item())

                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Average loss for epoch
                avg_loss = sum(epoch_losses) / len(epoch_losses)

                # Save training checkpoint
                checkpoint_mgr.save_training_checkpoint(
                    model, optimizer, epoch=epoch, step=(epoch + 1) * 10,
                    metrics={'loss': avg_loss}
                )

                # Save milestone if improved
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    checkpoint_mgr.save_milestone_checkpoint(
                        model, metric_value=avg_loss, metric_name='loss',
                        epoch=epoch, optimizer=optimizer
                    )

            # Verify checkpoints were created
            training_dir = Path(tmpdir) / 'checkpoints' / 'training'
            milestone_dir = Path(tmpdir) / 'checkpoints' / 'milestone'

            training_checkpoints = list(training_dir.glob('checkpoint_*.pt'))
            milestone_checkpoints = list(milestone_dir.glob('best_*.pt'))

            assert len(training_checkpoints) > 0, "No training checkpoints created"
            assert len(milestone_checkpoints) > 0, "No milestone checkpoints created"

            # Test resume from latest checkpoint
            latest_checkpoint = sorted(training_checkpoints)[-1]

            # Create new model and optimizer for resume
            new_model = BayesianExpectationTransformerLayer(config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

            # Load checkpoint
            state = checkpoint_mgr.load_checkpoint(
                str(latest_checkpoint), new_model, new_optimizer
            )

            assert state['epoch'] >= 0, "Invalid epoch in checkpoint"
            assert 'loss' in state['metrics'], "Missing loss metric"

            # Verify parameters match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(),
                new_model.named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-6), f"Parameter mismatch: {n1}"

    def test_uncertainty_quantification_workflow(self):
        """Test uncertainty metrics throughout workflow."""
        config = {
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': 1000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)

        # Generate batch
        x = torch.randn(4, 16, config['d_model'])

        # Forward pass with uncertainty
        output = model(x, return_uncertainty=True)

        # Check all uncertainty outputs
        assert 'uncertainty' in output, "Missing uncertainty output"
        uncertainty = output['uncertainty']

        required_keys = ['epistemic', 'aleatoric', 'total', 'posterior_alpha', 'posterior_beta']
        for key in required_keys:
            assert key in uncertainty, f"Missing uncertainty key: {key}"

        # Verify shapes
        batch_size = x.shape[0]
        assert uncertainty['epistemic'].shape[0] == batch_size
        assert uncertainty['aleatoric'].shape[0] == batch_size
        assert uncertainty['total'].shape[0] == batch_size

        # Verify ranges and properties
        assert (uncertainty['total'] >= 0).all(), "Total uncertainty must be non-negative"
        assert (uncertainty['epistemic'] >= 0).all(), "Epistemic uncertainty must be non-negative"
        assert (uncertainty['aleatoric'] >= 0).all(), "Aleatoric uncertainty must be non-negative"

        # Total should be sum of epistemic and aleatoric
        expected_total = uncertainty['epistemic'] + uncertainty['aleatoric']
        assert torch.allclose(uncertainty['total'], expected_total, atol=1e-6), \
            "Total uncertainty should equal epistemic + aleatoric"

        # Posterior parameters should be positive
        assert (uncertainty['posterior_alpha'] > 0).all(), "Alpha must be positive"
        assert (uncertainty['posterior_beta'] > 0).all(), "Beta must be positive"

    def test_cot_generation_workflow(self):
        """Test Chain-of-Thought generation in workflow."""
        config = {
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': 1000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)

        # Generate batch
        x = torch.randn(4, 16, config['d_model'])

        # Forward pass with CoT generation
        output = model(x, generate_cot=True)

        # Check CoT outputs
        assert 'cot_output' in output, "Missing CoT output"
        cot_output = output['cot_output']

        required_keys = ['optimal_lengths', 'reasoning_entropy', 'cot_logits', 'cot_tokens']
        for key in required_keys:
            assert key in cot_output, f"Missing CoT key: {key}"

        # Verify shapes
        batch_size, seq_length = x.shape[0], x.shape[1]

        assert cot_output['optimal_lengths'].shape == (batch_size,)
        assert cot_output['reasoning_entropy'].shape == (batch_size,)
        assert cot_output['cot_logits'].shape == (batch_size, seq_length, config['vocab_size'])
        assert cot_output['cot_tokens'].shape == (batch_size, seq_length)

        # Verify optimal lengths are reasonable
        assert (cot_output['optimal_lengths'] > 0).all(), "Optimal lengths must be positive"
        assert (cot_output['optimal_lengths'] <= 512).all(), "Optimal lengths exceed max"

        # Verify reasoning entropy is positive
        assert (cot_output['reasoning_entropy'] > 0).all(), "Reasoning entropy must be positive"

        # Verify token predictions are valid
        assert (cot_output['cot_tokens'] >= 0).all(), "Invalid token indices"
        assert (cot_output['cot_tokens'] < config['vocab_size']).all(), \
            "Token indices exceed vocab size"

    def test_end_to_end_gradient_flow(self):
        """Test that gradients flow through all components in full workflow."""
        config = {
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': 1000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training step
        x = torch.randn(4, 16, config['d_model'], requires_grad=True)
        output = model(x, return_uncertainty=True, generate_cot=True)

        # Compute loss using all outputs
        loss = output['hidden_states'].mean()

        # Add uncertainty regularization
        if 'uncertainty' in output:
            loss += 0.01 * output['uncertainty']['total'].mean()

        # Backward pass
        loss.backward()

        # Verify all parameters have gradients
        params_with_grad = 0
        params_without_grad = []

        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isnan(param.grad).any():
                params_with_grad += 1
            else:
                params_without_grad.append(name)

        assert len(params_without_grad) == 0, \
            f"Parameters without gradients: {params_without_grad}"

        # Optimizer step should not raise errors
        optimizer.step()
        optimizer.zero_grad()

    def test_production_deployment_workflow(self):
        """Test production checkpoint creation and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'd_model': 64,
                'n_heads': 4,
                'vocab_size': 1000,
                'k_permutations': 5,
                'dropout': 0.1
            }

            model = BayesianExpectationTransformerLayer(config)

            from src.bayesian_transformer.checkpointing import CheckpointManager
            checkpoint_mgr = CheckpointManager(checkpoint_dir=f'{tmpdir}/checkpoints')

            # Save production checkpoint
            metrics = {'accuracy': 0.89, 'loss': 0.25}
            version = '1.0.0'

            checkpoint_path = checkpoint_mgr.save_production_checkpoint(
                model, version=version, config=config, metrics=metrics,
                model_card={'description': 'Test model', 'task': 'testing'}
            )

            assert Path(checkpoint_path).exists(), "Production checkpoint not created"

            # Load for inference
            inference_model = BayesianExpectationTransformerLayer(config)
            state = checkpoint_mgr.load_checkpoint(checkpoint_path, inference_model)

            assert state['version'] == version, "Version mismatch"
            assert state['metrics'] == metrics, "Metrics mismatch"

            # Test inference
            x = torch.randn(2, 16, config['d_model'])

            with torch.no_grad():
                output = inference_model(x)

            assert 'hidden_states' in output, "Missing output from loaded model"
            assert output['hidden_states'].shape == (2, 16, config['d_model'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
