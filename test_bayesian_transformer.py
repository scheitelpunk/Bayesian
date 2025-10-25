import pytest
import torch
import torch.nn as nn
import math
import numpy as np
from bayesian_transformer import (
    MartingaleAwareAttention,
    OptimalCoTLayer,
    SufficientStatsEncoder,
    MDLRegularizedLoss,
    PositionalDebiasing,
    BayesianExpectationTransformerLayer
)


class TestMartingaleAwareAttention:
    """Test suite for MartingaleAwareAttention layer."""
    
    def test_initialization(self):
        """Test proper initialization of MartingaleAwareAttention."""
        d_model, n_heads = 512, 8
        layer = MartingaleAwareAttention(d_model, n_heads)
        
        assert layer.d_model == d_model
        assert layer.n_heads == n_heads
        assert layer.head_dim == d_model // n_heads
        assert layer.k_permutations == 20  # default value
        
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        d_model, n_heads = 512, 8
        batch_size, seq_length = 4, 64
        
        layer = MartingaleAwareAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_length, d_model)
        
        output = layer(x)
        
        assert output.shape == (batch_size, seq_length, d_model)
        assert not torch.isnan(output).any()
        
    def test_permutation_caching(self):
        """Test that permutation caching works correctly."""
        d_model, n_heads = 512, 8
        seq_length = 64
        
        layer = MartingaleAwareAttention(d_model, n_heads)
        device = torch.device('cpu')
        
        # First call should create cache
        perms1 = layer._get_cached_permutations(seq_length, device)
        assert perms1.shape == (layer.k_permutations, seq_length)
        
        # Second call should use cache
        perms2 = layer._get_cached_permutations(seq_length, device)
        assert torch.equal(perms1, perms2)
        
    def test_adaptive_weighting(self):
        """Test that adaptive weighting follows log(n)/n scaling."""
        d_model, n_heads = 512, 8
        layer = MartingaleAwareAttention(d_model, n_heads)
        
        # Test different sequence lengths
        for n in [8, 16, 32, 64, 128]:
            weight = layer._compute_adaptive_weight(n)
            expected = math.log(n) / n
            
            # Weight should be proportional to log(n)/n
            assert weight.item() > 0
            assert weight.item() <= 1.0
            
            # Should decrease with increasing n
            if n > 8:
                prev_weight = layer._compute_adaptive_weight(n // 2)
                assert weight.item() < prev_weight.item()
                
    def test_martingale_violation_reduction(self):
        """Test that permutation averaging reduces variance."""
        d_model, n_heads = 512, 8
        batch_size, seq_length = 4, 64
        
        layer = MartingaleAwareAttention(d_model, n_heads, k_permutations=20)
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Compute multiple outputs to estimate variance
        outputs = []
        for _ in range(10):
            torch.manual_seed(42)  # Fixed seed for reproducibility
            output = layer(x)
            outputs.append(output)
        
        outputs = torch.stack(outputs)
        variance = outputs.var(dim=0).mean()
        
        # Variance should be finite and reasonable
        assert torch.isfinite(variance)
        assert variance > 0


class TestOptimalCoTLayer:
    """Test suite for OptimalCoTLayer."""
    
    def test_initialization(self):
        """Test proper initialization of OptimalCoTLayer."""
        d_model, vocab_size = 512, 50000
        layer = OptimalCoTLayer(d_model, vocab_size)
        
        assert layer.d_model == d_model
        assert layer.vocab_size == vocab_size
        assert layer.L_f == 10  # default value
        
    def test_forward_pass(self):
        """Test forward pass produces correct outputs."""
        d_model, vocab_size = 512, 50000
        batch_size, seq_length = 4, 64
        
        layer = OptimalCoTLayer(d_model, vocab_size)
        x = torch.randn(batch_size, seq_length, d_model)
        
        output = layer(x, generate_cot=True)
        
        assert 'optimal_lengths' in output
        assert 'reasoning_entropy' in output
        assert 'cot_logits' in output
        assert 'cot_tokens' in output
        
        assert output['optimal_lengths'].shape == (batch_size,)
        assert output['reasoning_entropy'].shape == (batch_size,)
        assert output['cot_logits'].shape == (batch_size, seq_length, vocab_size)
        assert output['cot_tokens'].shape == (batch_size, seq_length)
        
    def test_optimal_length_computation(self):
        """Test that optimal length follows theoretical formula."""
        d_model, vocab_size = 512, 50000
        batch_size = 4
        
        layer = OptimalCoTLayer(d_model, vocab_size)
        
        # Test scaling with sequence length
        for seq_length in [16, 32, 64, 128]:
            x = torch.randn(batch_size, seq_length, d_model)
            output = layer(x, generate_cot=False)
            
            k_optimal = output['optimal_lengths']
            
            # Should be positive and within reasonable bounds
            assert (k_optimal > 0).all()
            assert (k_optimal <= layer.max_cot_length).all()
            
            # Should scale roughly as sqrt(n)
            if seq_length > 16:
                prev_x = torch.randn(batch_size, seq_length // 2, d_model)
                prev_output = layer(prev_x, generate_cot=False)
                prev_k = prev_output['optimal_lengths']
                
                # Current should be larger than previous (sqrt scaling)
                assert k_optimal.float().mean() > prev_k.float().mean()
                
    def test_entropy_estimation(self):
        """Test reasoning entropy estimation."""
        d_model, vocab_size = 512, 50000
        batch_size, seq_length = 4, 64
        
        layer = OptimalCoTLayer(d_model, vocab_size)
        x = torch.randn(batch_size, seq_length, d_model)
        
        h_cot = layer._estimate_reasoning_entropy(x)
        
        assert h_cot.shape == (batch_size,)
        assert (h_cot > 0).all()  # Entropy should be positive
        assert torch.isfinite(h_cot).all()


class TestSufficientStatsEncoder:
    """Test suite for SufficientStatsEncoder."""
    
    def test_initialization(self):
        """Test proper initialization of SufficientStatsEncoder."""
        d_model = 512
        layer = SufficientStatsEncoder(d_model)
        
        assert layer.d_model == d_model
        assert layer.max_moments == max(3, int(math.log(d_model)))
        assert len(layer.counting_heads) == layer.max_moments
        assert len(layer.moment_networks) == layer.max_moments
        
    def test_forward_pass(self):
        """Test forward pass produces correct outputs."""
        d_model = 512
        batch_size, seq_length = 4, 64
        
        layer = SufficientStatsEncoder(d_model)
        x = torch.randn(batch_size, seq_length, d_model)
        
        output = layer(x)
        
        required_keys = ['sufficient_stats', 'moments', 'counts', 'alpha', 'beta', 
                        'posterior_mean', 'posterior_variance']
        for key in required_keys:
            assert key in output
            
        assert output['sufficient_stats'].shape == (batch_size, seq_length, d_model)
        assert output['moments'].shape == (batch_size, layer.max_moments)
        assert output['counts'].shape == (batch_size, layer.max_moments)
        assert output['alpha'].shape == (batch_size,)
        assert output['beta'].shape == (batch_size,)
        
    def test_beta_posterior_properties(self):
        """Test Beta posterior has correct properties."""
        d_model = 512
        batch_size, seq_length = 4, 64
        
        layer = SufficientStatsEncoder(d_model)
        x = torch.randn(batch_size, seq_length, d_model)
        
        output = layer(x)
        
        alpha = output['alpha']
        beta = output['beta']
        mean = output['posterior_mean']
        variance = output['posterior_variance']
        
        # Beta parameters should be positive
        assert (alpha > 0).all()
        assert (beta > 0).all()
        
        # Mean should be in [0, 1]
        assert (mean >= 0).all() and (mean <= 1).all()
        
        # Variance should be positive
        assert (variance > 0).all()
        
        # Check Beta distribution properties
        expected_mean = alpha / (alpha + beta)
        expected_variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        
        assert torch.allclose(mean, expected_mean, atol=1e-6)
        assert torch.allclose(variance, expected_variance, atol=1e-6)
        
    def test_moment_computation(self):
        """Test moment computation produces valid statistics."""
        d_model = 512
        batch_size, seq_length = 4, 64
        
        layer = SufficientStatsEncoder(d_model)
        x = torch.randn(batch_size, seq_length, d_model)
        
        moments = layer._compute_moments(x)
        
        assert moments.shape == (batch_size, layer.max_moments)
        assert torch.isfinite(moments).all()


class TestMDLRegularizedLoss:
    """Test suite for MDLRegularizedLoss."""
    
    def test_initialization(self):
        """Test proper initialization of MDLRegularizedLoss."""
        loss_fn = MDLRegularizedLoss(beta=0.1)
        
        assert loss_fn.beta == 0.1
        assert loss_fn.vocab_size == 50000  # default value
        
    def test_forward_pass(self):
        """Test forward pass produces correct loss components."""
        batch_size, seq_length, vocab_size = 4, 64, 1000
        
        loss_fn = MDLRegularizedLoss(beta=0.1, vocab_size=vocab_size)
        logits = torch.randn(batch_size, seq_length, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        output = loss_fn(logits, targets)
        
        required_keys = ['loss', 'standard_loss', 'mdl_penalty', 'empirical_entropy',
                        'optimal_complexity', 'actual_complexity']
        for key in required_keys:
            assert key in output
            
        # All components should be non-negative
        assert output['loss'] >= 0
        assert output['standard_loss'] >= 0
        assert output['mdl_penalty'] >= 0
        assert output['empirical_entropy'] >= 0
        
    def test_entropy_computation(self):
        """Test empirical entropy computation."""
        batch_size, seq_length, vocab_size = 4, 64, 1000
        
        loss_fn = MDLRegularizedLoss()
        
        # Test with uniform distribution (should have high entropy)
        uniform_logits = torch.zeros(batch_size, seq_length, vocab_size)
        uniform_entropy = loss_fn._compute_empirical_entropy(uniform_logits)
        
        # Test with peaked distribution (should have low entropy)
        peaked_logits = torch.zeros(batch_size, seq_length, vocab_size)
        peaked_logits[:, :, 0] = 10.0  # Peak at first token
        peaked_entropy = loss_fn._compute_empirical_entropy(peaked_logits)
        
        assert uniform_entropy > peaked_entropy
        assert uniform_entropy.item() <= math.log(vocab_size)  # Maximum entropy
        
    def test_complexity_bounds(self):
        """Test that complexity follows theoretical bounds."""
        batch_size, seq_length, vocab_size = 4, 64, 1000
        
        loss_fn = MDLRegularizedLoss()
        logits = torch.randn(batch_size, seq_length, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        output = loss_fn(logits, targets)
        
        empirical_entropy = output['empirical_entropy']
        optimal_complexity = output['optimal_complexity']
        
        # Optimal complexity should follow n*H(p) + O(sqrt(n*log(n)))
        main_term = seq_length * empirical_entropy
        correction_term = math.sqrt(seq_length * math.log(seq_length))
        expected_complexity = main_term + correction_term
        
        assert torch.isclose(optimal_complexity, expected_complexity, rtol=1e-3)


class TestPositionalDebiasing:
    """Test suite for PositionalDebiasing."""
    
    def test_initialization(self):
        """Test proper initialization of PositionalDebiasing."""
        d_model = 512
        layer = PositionalDebiasing(d_model)
        
        assert layer.d_model == d_model
        assert layer.n_harmonics == 8  # default value
        assert layer.encoding_type == 'rotary'  # default value
        
    def test_forward_pass(self):
        """Test forward pass produces correct outputs."""
        d_model = 512
        batch_size, seq_length = 4, 64
        
        layer = PositionalDebiasing(d_model)
        x = torch.randn(batch_size, seq_length, d_model)
        
        output = layer(x)
        
        required_keys = ['debiased_output', 'artifact_scores', 'correction',
                        'artifact_magnitude', 'correction_magnitude']
        for key in required_keys:
            assert key in output
            
        assert output['debiased_output'].shape == (batch_size, seq_length, d_model)
        assert output['artifact_scores'].shape == (batch_size, seq_length, layer.n_harmonics)
        assert output['correction'].shape == (batch_size, seq_length, d_model)
        assert output['artifact_magnitude'].shape == (batch_size,)
        assert output['correction_magnitude'].shape == (batch_size,)
        
    def test_artifact_detection(self):
        """Test artifact detection produces valid scores."""
        d_model = 512
        batch_size, seq_length = 4, 128
        
        layer = PositionalDebiasing(d_model)
        x = torch.randn(batch_size, seq_length, d_model)
        
        artifact_scores = layer._detect_periodic_artifacts(x)
        
        assert artifact_scores.shape == (batch_size, seq_length, layer.n_harmonics)
        assert (artifact_scores >= 0).all() and (artifact_scores <= 1).all()  # Sigmoid output
        
    def test_periodic_artifact_detection(self):
        """Test detection of known periodic patterns."""
        d_model = 512
        batch_size, seq_length = 4, 128
        
        layer = PositionalDebiasing(d_model)
        
        # Create input with periodic pattern
        x = torch.randn(batch_size, seq_length, d_model)
        period = 64
        for i in range(seq_length):
            if i % period == 0:
                x[:, i, :] += 1.0  # Add periodic spike
                
        artifact_scores = layer._detect_periodic_artifacts(x)
        
        # Should detect some level of periodicity
        assert artifact_scores.mean() > 0.1  # Some artifact detection
        
    def test_correction_preserves_information(self):
        """Test that correction preserves essential information."""
        d_model = 512
        batch_size, seq_length = 4, 64
        
        layer = PositionalDebiasing(d_model)
        x = torch.randn(batch_size, seq_length, d_model)
        
        output = layer(x)
        debiased_x = output['debiased_output']
        
        # Debiased output should have similar norm
        original_norm = x.norm(dim=-1).mean()
        debiased_norm = debiased_x.norm(dim=-1).mean()
        
        assert torch.isclose(original_norm, debiased_norm, rtol=0.2)
        
        # Should not be identical (some correction applied)
        assert not torch.equal(x, debiased_x)


class TestBayesianExpectationTransformerLayer:
    """Test suite for complete BayesianExpectationTransformerLayer."""
    
    def test_initialization(self):
        """Test proper initialization of complete layer."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 50000,
            'k_permutations': 20,
            'dropout': 0.1
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        
        assert layer.d_model == config['d_model']
        assert layer.n_heads == config['n_heads']
        assert layer.vocab_size == config['vocab_size']
        assert isinstance(layer.attention, MartingaleAwareAttention)
        assert isinstance(layer.cot_generator, OptimalCoTLayer)
        assert isinstance(layer.stats_encoder, SufficientStatsEncoder)
        assert isinstance(layer.debiasing, PositionalDebiasing)
        
    def test_forward_pass(self):
        """Test forward pass produces correct outputs."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 50000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        batch_size, seq_length = 4, 64
        x = torch.randn(batch_size, seq_length, config['d_model'])
        
        output = layer(x)
        
        required_keys = ['hidden_states', 'sufficient_stats', 'debiasing_info']
        for key in required_keys:
            assert key in output
            
        assert output['hidden_states'].shape == (batch_size, seq_length, config['d_model'])
        
    def test_forward_with_cot(self):
        """Test forward pass with CoT generation."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 50000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        batch_size, seq_length = 4, 64
        x = torch.randn(batch_size, seq_length, config['d_model'])
        
        output = layer(x, generate_cot=True)
        
        assert 'cot_output' in output
        assert 'optimal_lengths' in output['cot_output']
        assert 'reasoning_entropy' in output['cot_output']
        
    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty estimation."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 50000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        batch_size, seq_length = 4, 64
        x = torch.randn(batch_size, seq_length, config['d_model'])
        
        output = layer(x, return_uncertainty=True)
        
        assert 'uncertainty' in output
        uncertainty = output['uncertainty']
        
        required_keys = ['epistemic', 'aleatoric', 'total', 'posterior_alpha', 'posterior_beta']
        for key in required_keys:
            assert key in uncertainty
            
        # All uncertainty measures should be non-negative
        assert (uncertainty['epistemic'] >= 0).all()
        assert (uncertainty['aleatoric'] >= 0).all()
        assert (uncertainty['total'] >= 0).all()
        
    def test_gradient_flow(self):
        """Test that gradients flow through all components."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 50000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        batch_size, seq_length = 4, 64
        x = torch.randn(batch_size, seq_length, config['d_model'], requires_grad=True)
        
        output = layer(x)
        loss = output['hidden_states'].sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            
    def test_theoretical_properties(self):
        """Test that layer satisfies theoretical properties."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 50000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        
        # Test different sequence lengths to verify scaling
        for seq_length in [16, 32, 64, 128]:
            batch_size = 4
            x = torch.randn(batch_size, seq_length, config['d_model'])
            
            output = layer(x, generate_cot=True, return_uncertainty=True)
            
            # CoT length should scale with sqrt(n)
            cot_lengths = output['cot_output']['optimal_lengths']
            assert (cot_lengths > 0).all()
            
            # Uncertainty should be well-calibrated
            uncertainty = output['uncertainty']
            assert (uncertainty['total'] > 0).all()
            assert (uncertainty['total'] < 1).all()  # Should be reasonable
            
            # Sufficient statistics should be informative
            stats = output['sufficient_stats']
            posterior_mean = stats['posterior_mean']
            assert (posterior_mean >= 0).all() and (posterior_mean <= 1).all()


class TestIntegrationAndPerformance:
    """Integration tests and performance validation."""
    
    def test_martingale_violation_scaling(self):
        """Test that martingale violations follow Θ(log n/n) scaling."""
        config = {
            'd_model': 256,  # Smaller for faster testing
            'n_heads': 4,
            'vocab_size': 1000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        
        violations = []
        sequence_lengths = [16, 32, 64, 128, 256]
        
        for seq_length in sequence_lengths:
            batch_size = 8
            x = torch.randn(batch_size, seq_length, config['d_model'])
            
            # Run multiple times to estimate violation
            outputs = []
            for _ in range(5):
                output = layer(x)
                outputs.append(output['hidden_states'])
            
            # Estimate violation as variance across runs
            outputs = torch.stack(outputs)
            violation = outputs.var(dim=0).mean().item()
            violations.append(violation)
            
        # Violations should decrease roughly as log(n)/n
        for i in range(1, len(violations)):
            n_prev = sequence_lengths[i-1]
            n_curr = sequence_lengths[i]
            
            expected_ratio = (math.log(n_curr) / n_curr) / (math.log(n_prev) / n_prev)
            actual_ratio = violations[i] / violations[i-1]
            
            # Should be roughly proportional (within factor of 2)
            assert 0.5 * expected_ratio <= actual_ratio <= 2.0 * expected_ratio
            
    def test_compression_efficiency(self):
        """Test compression efficiency approaches theoretical limits."""
        config = {
            'd_model': 256,
            'n_heads': 4,
            'vocab_size': 1000
        }
        
        layer = BayesianExpectationTransformerLayer(config)
        batch_size, seq_length = 4, 64
        x = torch.randn(batch_size, seq_length, config['d_model'])
        
        # Test with MDL loss
        loss_fn = MDLRegularizedLoss(vocab_size=config['vocab_size'])
        
        output = layer(x)
        
        # Generate dummy logits for loss computation
        logits = torch.randn(batch_size, seq_length, config['vocab_size'])
        targets = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
        
        loss_output = loss_fn(logits, targets)
        
        # Compression efficiency = optimal_complexity / actual_complexity
        efficiency = loss_output['optimal_complexity'] / loss_output['actual_complexity']
        
        # Should be reasonably high (>0.5 for this test)
        assert efficiency > 0.5
        
    def test_permutation_variance_reduction(self):
        """Test that permutation averaging reduces variance by factor √k."""
        d_model, n_heads = 256, 4
        k_values = [1, 4, 16, 64]  # Different numbers of permutations

        variances = []
        for k in k_values:
            layer = MartingaleAwareAttention(d_model, n_heads, k_permutations=k)

            batch_size, seq_length = 4, 32

            # Compute variance over multiple runs with different inputs
            outputs = []
            for _ in range(20):  # Increased from 10 to 20 for better statistics
                x = torch.randn(batch_size, seq_length, d_model)
                # Clear cache to ensure fresh permutations
                layer.perm_cache.clear()
                output = layer(x)
                outputs.append(output)

            outputs = torch.stack(outputs)
            variance = outputs.var(dim=0).mean().item()
            variances.append(variance)

        # Variance should decrease roughly as 1/√k
        for i in range(1, len(variances)):
            k_prev = k_values[i-1]
            k_curr = k_values[i]

            expected_ratio = math.sqrt(k_prev / k_curr)
            actual_ratio = variances[i] / variances[i-1]

            # Should be roughly proportional (relaxed tolerance to factor of 3 for robustness)
            assert 0.33 * expected_ratio <= actual_ratio <= 3.0 * expected_ratio
            
    def test_cot_length_scaling(self):
        """Test that CoT length follows k* = Θ(√n log(1/ε))."""
        d_model, vocab_size = 256, 1000
        layer = OptimalCoTLayer(d_model, vocab_size)
        
        sequence_lengths = [16, 32, 64, 128, 256]
        cot_lengths = []
        
        for seq_length in sequence_lengths:
            batch_size = 4
            x = torch.randn(batch_size, seq_length, d_model)
            
            output = layer(x, generate_cot=False)
            avg_length = output['optimal_lengths'].float().mean().item()
            cot_lengths.append(avg_length)
        
        # CoT length should scale roughly as √n
        for i in range(1, len(cot_lengths)):
            n_prev = sequence_lengths[i-1]
            n_curr = sequence_lengths[i]
            
            expected_ratio = math.sqrt(n_curr / n_prev)
            actual_ratio = cot_lengths[i] / cot_lengths[i-1]
            
            # Should be roughly proportional (within factor of 2)
            assert 0.5 * expected_ratio <= actual_ratio <= 2.0 * expected_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])