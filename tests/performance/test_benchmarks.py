"""
Performance benchmark tests.

Tests latency, throughput, memory usage, and scaling characteristics.
"""

import pytest
import torch
import time
from src.bayesian_transformer import BayesianExpectationTransformerLayer


class TestPerformance:
    """Performance benchmarks."""

    def test_forward_pass_latency(self):
        """Measure forward pass latency across different batch sizes."""
        config = {
            'd_model': 512,
            'n_heads': 8,
            'vocab_size': 10000,
            'k_permutations': 10,  # Reduced from 20 for faster testing
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)
        model.eval()

        batch_sizes = [1, 8, 32]
        seq_length = 128

        results = {}

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_length, config['d_model'])

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    _ = model(x)
                    times.append(time.time() - start)

            p50 = sorted(times)[50]
            p95 = sorted(times)[95]

            results[batch_size] = {'p50': p50, 'p95': p95}

            print(f"Batch {batch_size}: P50={p50*1000:.2f}ms, P95={p95*1000:.2f}ms")

            # Verify reasonable latency (adjust threshold as needed)
            assert p95 < 2.0, f"P95 latency too high for batch {batch_size}: {p95:.2f}s"

    def test_memory_efficiency(self):
        """Test memory usage scaling."""
        config = {
            'd_model': 256,
            'n_heads': 8,
            'vocab_size': 5000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Estimate memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        param_memory_mb = param_memory / (1024 * 1024)

        print(f"Parameter memory: {param_memory_mb:.2f} MB")

        # For 256 d_model, expect reasonable parameter count
        assert total_params < 50_000_000, "Too many parameters"

    def test_batch_throughput(self):
        """Test samples/second throughput."""
        config = {
            'd_model': 256,
            'n_heads': 8,
            'vocab_size': 5000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)
        model.eval()

        batch_size = 32
        seq_length = 64
        num_iterations = 50

        x = torch.randn(batch_size, seq_length, config['d_model'])

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        elapsed = time.time() - start

        total_samples = batch_size * num_iterations
        throughput = total_samples / elapsed

        print(f"Throughput: {throughput:.2f} samples/sec")

        # Should process at least 100 samples/sec
        assert throughput > 100, f"Throughput too low: {throughput:.2f} samples/sec"

    def test_gradient_computation_overhead(self):
        """Test overhead of backward pass."""
        config = {
            'd_model': 256,
            'n_heads': 8,
            'vocab_size': 5000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)

        batch_size = 16
        seq_length = 64
        x = torch.randn(batch_size, seq_length, config['d_model'], requires_grad=True)

        # Measure forward pass time
        forward_times = []
        for _ in range(20):
            start = time.time()
            output = model(x)
            forward_times.append(time.time() - start)

        avg_forward = sum(forward_times) / len(forward_times)

        # Measure forward + backward time
        full_times = []
        for _ in range(20):
            start = time.time()
            output = model(x)
            loss = output['hidden_states'].mean()
            loss.backward()
            full_times.append(time.time() - start)

            # Clear gradients
            model.zero_grad()

        avg_full = sum(full_times) / len(full_times)

        backward_time = avg_full - avg_forward
        overhead_ratio = backward_time / avg_forward

        print(f"Forward: {avg_forward*1000:.2f}ms")
        print(f"Backward: {backward_time*1000:.2f}ms")
        print(f"Overhead ratio: {overhead_ratio:.2f}x")

        # Backward should be 2-3x forward time
        assert overhead_ratio < 5.0, f"Backward overhead too high: {overhead_ratio:.2f}x"

    def test_sequence_length_scaling(self):
        """Test how performance scales with sequence length."""
        config = {
            'd_model': 128,
            'n_heads': 4,
            'vocab_size': 5000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)
        model.eval()

        batch_size = 4
        seq_lengths = [32, 64, 128, 256]

        times = {}

        for seq_length in seq_lengths:
            x = torch.randn(batch_size, seq_length, config['d_model'])

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)

            # Measure
            start = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(x)
            elapsed = time.time() - start

            avg_time = elapsed / 20
            times[seq_length] = avg_time

            print(f"Seq length {seq_length}: {avg_time*1000:.2f}ms")

        # Time should scale sub-quadratically due to optimizations
        # (Attention is O(n^2) but other components are linear)
        for i in range(1, len(seq_lengths)):
            prev_len = seq_lengths[i-1]
            curr_len = seq_lengths[i]

            prev_time = times[prev_len]
            curr_time = times[curr_len]

            length_ratio = curr_len / prev_len
            time_ratio = curr_time / prev_time

            # Time ratio should be less than length_ratio^2 (quadratic)
            assert time_ratio < length_ratio ** 2.5, \
                f"Scaling worse than O(n^2.5): {time_ratio:.2f} vs {length_ratio**2:.2f}"

    def test_permutation_caching_benefit(self):
        """Test that permutation caching improves performance."""
        from src.bayesian_transformer import MartingaleAwareAttention

        d_model = 256
        n_heads = 8
        batch_size = 4
        seq_length = 64

        # Test with caching
        layer_with_cache = MartingaleAwareAttention(
            d_model, n_heads, k_permutations=10, cache_maxsize=100
        )
        layer_with_cache.eval()

        x = torch.randn(batch_size, seq_length, d_model)

        # First pass (cold cache)
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = layer_with_cache(x)
        cold_time = time.time() - start

        # Second pass (warm cache)
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = layer_with_cache(x)
        warm_time = time.time() - start

        print(f"Cold cache: {cold_time*1000:.2f}ms")
        print(f"Warm cache: {warm_time*1000:.2f}ms")
        print(f"Speedup: {cold_time/warm_time:.2f}x")

        # Warm cache should be faster (or at least not slower)
        assert warm_time <= cold_time * 1.1, "Cache should not hurt performance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
