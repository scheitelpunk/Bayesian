"""
Integration tests for IMDB data pipeline.

Tests data loading, tokenization, and model integration with real IMDB reviews.
"""

import pytest
import torch


class TestIMDBPipeline:
    """Tests for IMDB data integration."""

    def test_imdb_dataset_loads(self):
        """Verify IMDB dataset can be loaded."""
        pytest.importorskip("datasets")
        from datasets import load_dataset

        # Load small subset for testing
        dataset = load_dataset('stanfordnlp/imdb', split='test', streaming=True)

        # Get first sample
        sample = next(iter(dataset))

        assert 'text' in sample, "Missing 'text' field in IMDB sample"
        assert 'label' in sample, "Missing 'label' field in IMDB sample"
        assert isinstance(sample['text'], str), "Text should be string"
        assert sample['label'] in [0, 1], "Label should be 0 or 1"
        assert len(sample['text']) > 0, "Text should not be empty"

    def test_simple_tokenization(self):
        """Test basic tokenization pipeline."""
        texts = [
            "This movie was great!",
            "Terrible film, waste of time.",
            "Average movie, nothing special."
        ]

        # Simple word-level tokenizer for testing
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())

        vocab = {word: idx for idx, word in enumerate(sorted(vocab))}
        vocab['<PAD>'] = len(vocab)
        vocab['<UNK>'] = len(vocab) + 1

        assert len(vocab) > 0, "Vocabulary should not be empty"
        assert '<PAD>' in vocab, "Missing PAD token"
        assert '<UNK>' in vocab, "Missing UNK token"

        # Tokenize texts
        max_length = 10
        tokenized = []

        for text in texts:
            tokens = text.lower().split()[:max_length]
            token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]

            # Pad to max length
            while len(token_ids) < max_length:
                token_ids.append(vocab['<PAD>'])

            tokenized.append(token_ids)

        tokenized_tensor = torch.tensor(tokenized)

        assert tokenized_tensor.shape == (3, max_length), "Unexpected shape"
        assert (tokenized_tensor >= 0).all(), "Invalid token IDs"
        assert (tokenized_tensor < len(vocab)).all(), "Token IDs exceed vocab size"

    def test_model_with_tokenized_input(self):
        """Test model processes tokenized IMDB-like data."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer

        config = {
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': 1000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)

        # Simulate tokenized batch
        batch_size = 4
        seq_length = 32

        # Random token IDs
        token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length))

        # Simple embedding layer
        embedding = torch.nn.Embedding(config['vocab_size'], config['d_model'])
        embedded = embedding(token_ids)

        assert embedded.shape == (batch_size, seq_length, config['d_model'])

        # Forward pass through model
        output = model(embedded)

        assert 'hidden_states' in output, "Missing hidden states"
        assert output['hidden_states'].shape == (batch_size, seq_length, config['d_model'])

    def test_sentiment_classification_pipeline(self):
        """Test complete sentiment classification pipeline."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        import torch.nn.functional as F

        config = {
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': 1000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        # Build simple classifier
        transformer = BayesianExpectationTransformerLayer(config)
        embedding = torch.nn.Embedding(config['vocab_size'], config['d_model'])
        classifier = torch.nn.Linear(config['d_model'], 2)  # Binary classification

        # Simulate batch
        batch_size = 4
        seq_length = 32
        token_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
        labels = torch.randint(0, 2, (batch_size,))

        # Forward pass
        embedded = embedding(token_ids)
        transformer_out = transformer(embedded)

        # Pool sequence for classification (mean pooling)
        pooled = transformer_out['hidden_states'].mean(dim=1)
        logits = classifier(pooled)

        assert logits.shape == (batch_size, 2), "Unexpected logits shape"

        # Compute loss
        loss = F.cross_entropy(logits, labels)
        assert loss.item() > 0, "Loss should be positive"

        # Predictions
        predictions = torch.argmax(logits, dim=-1)
        assert predictions.shape == (batch_size,), "Unexpected predictions shape"
        assert (predictions >= 0).all() and (predictions < 2).all(), "Invalid predictions"

    def test_batch_processing_performance(self):
        """Test that batch processing works efficiently."""
        from src.bayesian_transformer import BayesianExpectationTransformerLayer
        import time

        config = {
            'd_model': 64,
            'n_heads': 4,
            'vocab_size': 1000,
            'k_permutations': 5,
            'dropout': 0.1
        }

        model = BayesianExpectationTransformerLayer(config)
        model.eval()

        batch_sizes = [1, 4, 8]
        seq_length = 32

        times = []

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_length, config['d_model'])

            # Warmup
            with torch.no_grad():
                _ = model(x)

            # Measure time
            start = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            elapsed = time.time() - start

            times.append(elapsed / 10)

        # Larger batches should have better throughput (time per sample)
        time_per_sample = [t / bs for t, bs in zip(times, batch_sizes)]

        # Batch 8 should be more efficient than batch 1
        assert time_per_sample[-1] < time_per_sample[0], \
            "Batching should improve efficiency"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
