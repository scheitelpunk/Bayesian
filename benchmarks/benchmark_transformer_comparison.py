"""
Comprehensive Benchmark: Bayesian Transformer vs Standard Transformer

Compares:
1. Training speed (time per epoch)
2. Inference speed (predictions per second)
3. Memory usage (GPU/CPU)
4. Model accuracy
5. Uncertainty quantification (Bayesian only)
6. Model size (parameters)
7. Convergence speed

Author: Bayesian Transformer Team
Date: 2025-10-25
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import psutil
import gc

from src.bayesian_transformer import BayesianExpectationTransformerLayer
from datasets import load_dataset


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    training_time_per_epoch: float  # seconds
    inference_time_per_sample: float  # milliseconds
    memory_usage_mb: float
    final_accuracy: float
    convergence_epoch: int  # epoch where accuracy > 80%
    total_parameters: int
    loss_history: List[float]
    accuracy_history: List[float]
    uncertainty_available: bool = False
    mean_uncertainty: float = 0.0
    uncertainty_correlation_with_errors: float = 0.0


class StandardTransformer(nn.Module):
    """Standard PyTorch Transformer for comparison."""

    def __init__(self, config: Dict):
        super().__init__()
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.vocab_size = config['vocab_size']
        self.dropout = config.get('dropout', 0.1)

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, self.d_model))

        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classification head
        self.classifier = nn.Linear(self.d_model, 2)  # Binary classification

    def forward(self, x, return_uncertainty=False):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape

        # Embedding
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(x)  # (batch, 2)

        if return_uncertainty:
            # Standard transformer has no uncertainty
            return {
                'logits': logits,
                'epistemic_uncertainty': torch.zeros(batch_size, device=x.device),
                'aleatoric_uncertainty': torch.zeros(batch_size, device=x.device)
            }
        return logits


class BayesianTransformerWrapper(nn.Module):
    """Wrapper for Bayesian Transformer with embedding layer."""

    def __init__(self, config: Dict):
        super().__init__()
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']

        # Embedding
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, self.d_model))

        # Bayesian Transformer Layer
        self.bayesian_layer = BayesianExpectationTransformerLayer(config)

        # Classification head
        self.classifier = nn.Linear(self.d_model, 2)  # Binary classification

    def forward(self, x, return_uncertainty=False):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.shape

        # Embedding
        x = self.embedding(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Bayesian Transformer
        outputs = self.bayesian_layer(x, return_uncertainty=return_uncertainty)

        # Extract hidden states
        if isinstance(outputs, dict):
            transformer_out = outputs['hidden_states']  # (batch, seq_len, d_model)
        else:
            transformer_out = outputs

        # Global average pooling
        pooled = transformer_out.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(pooled)  # (batch, 2)

        if return_uncertainty and isinstance(outputs, dict):
            # Extract uncertainty from the outputs
            uncertainty = outputs.get('uncertainty', {})
            epistemic = uncertainty.get('epistemic', torch.zeros(batch_size, seq_len, device=x.device))
            aleatoric = uncertainty.get('aleatoric', torch.zeros(batch_size, seq_len, device=x.device))

            # Average over sequence length
            epistemic_avg = epistemic.mean(dim=1) if epistemic.dim() > 1 else epistemic
            aleatoric_avg = aleatoric.mean(dim=1) if aleatoric.dim() > 1 else aleatoric

            return {
                'logits': logits,
                'epistemic_uncertainty': epistemic_avg,
                'aleatoric_uncertainty': aleatoric_avg
            }
        elif return_uncertainty:
            return {
                'logits': logits,
                'epistemic_uncertainty': torch.zeros(batch_size, device=x.device),
                'aleatoric_uncertainty': torch.zeros(batch_size, device=x.device)
            }
        return logits


class TransformerBenchmark:
    """Benchmark suite for transformer comparison."""

    def __init__(self, device='cpu', max_samples=1000):
        self.device = device
        self.max_samples = max_samples
        self.results: Dict[str, BenchmarkResult] = {}

    def load_imdb_data(self, max_samples=1000):
        """Load IMDB dataset for benchmarking."""
        print(f"Loading IMDB data (max {max_samples} samples)...")

        train_dataset = load_dataset('imdb', split='train', streaming=True)
        test_dataset = load_dataset('imdb', split='test', streaming=True)

        # Simple tokenizer
        def tokenize(text, vocab_size=10000, max_len=128):
            # Very basic tokenization
            words = text.lower().split()
            tokens = [hash(word) % vocab_size for word in words]
            if len(tokens) < max_len:
                tokens.extend([0] * (max_len - len(tokens)))
            else:
                tokens = tokens[:max_len]
            return tokens

        # Collect train data
        train_texts, train_labels = [], []
        for i, item in enumerate(train_dataset):
            if i >= max_samples:
                break
            train_texts.append(tokenize(item['text']))
            train_labels.append(item['label'])

        # Collect test data
        test_texts, test_labels = [], []
        for i, item in enumerate(test_dataset):
            if i >= max_samples // 5:  # 20% test size
                break
            test_texts.append(tokenize(item['text']))
            test_labels.append(item['label'])

        # Convert to tensors
        train_x = torch.tensor(train_texts, dtype=torch.long)
        train_y = torch.tensor(train_labels, dtype=torch.long)
        test_x = torch.tensor(test_texts, dtype=torch.long)
        test_y = torch.tensor(test_labels, dtype=torch.long)

        print(f"Loaded {len(train_x)} train samples, {len(test_x)} test samples")

        return (train_x, train_y), (test_x, test_y)

    def measure_memory(self) -> float:
        """Measure current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_and_evaluate(
        self,
        model: nn.Module,
        model_name: str,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        test_data: Tuple[torch.Tensor, torch.Tensor],
        n_epochs: int = 5,
        batch_size: int = 8,
        lr: float = 1e-4
    ) -> BenchmarkResult:
        """Train and evaluate a model, collecting benchmark metrics."""

        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")

        train_x, train_y = train_data
        test_x, test_y = test_data

        # Create dataloaders
        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Setup training
        model = model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # Count parameters
        total_params = self.count_parameters(model)
        print(f"Total parameters: {total_params:,}")

        # Measure initial memory
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        mem_before = self.measure_memory()

        # Training metrics
        loss_history = []
        accuracy_history = []
        epoch_times = []
        convergence_epoch = n_epochs  # Default to last epoch

        # Training loop
        for epoch in range(n_epochs):
            epoch_start = time.time()

            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)

                # Handle dict output (Bayesian) vs tensor output (Standard)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs

                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            epoch_loss /= len(train_loader)
            epoch_acc = correct / total

            loss_history.append(epoch_loss)
            accuracy_history.append(epoch_acc)

            # Check convergence
            if epoch_acc >= 0.80 and convergence_epoch == n_epochs:
                convergence_epoch = epoch + 1

            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"Loss = {epoch_loss:.4f}, "
                  f"Accuracy = {epoch_acc:.4f}, "
                  f"Time = {epoch_time:.2f}s")

        # Measure memory after training
        mem_after = self.measure_memory()
        memory_used = mem_after - mem_before

        # Evaluation
        print("\nEvaluating on test set...")
        model.eval()
        correct = 0
        total = 0
        all_uncertainties = []
        all_errors = []

        inference_times = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Measure inference time
                start = time.time()
                outputs = model(batch_x, return_uncertainty=True)
                inference_time = (time.time() - start) * 1000  # Convert to ms
                inference_times.append(inference_time / batch_x.size(0))  # Per sample

                # Handle dict output
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    uncertainties = outputs['epistemic_uncertainty'].cpu().numpy()
                else:
                    logits = outputs
                    uncertainties = np.zeros(batch_y.size(0))

                _, predicted = torch.max(logits.data, 1)
                errors = (predicted != batch_y).cpu().numpy()

                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                all_uncertainties.extend(uncertainties)
                all_errors.extend(errors)

        final_accuracy = correct / total
        avg_inference_time = np.mean(inference_times)

        print(f"\nTest Accuracy: {final_accuracy:.4f}")
        print(f"Avg inference time: {avg_inference_time:.2f} ms/sample")

        # Uncertainty analysis (for Bayesian models)
        uncertainty_available = np.any(all_uncertainties)
        uncertainty_correlation = 0.0
        mean_uncertainty = 0.0

        if uncertainty_available:
            mean_uncertainty = np.mean(all_uncertainties)
            # Correlation between uncertainty and errors
            if len(all_uncertainties) > 0 and len(all_errors) > 0:
                uncertainty_correlation = np.corrcoef(all_uncertainties, all_errors)[0, 1]
            print(f"Mean uncertainty: {mean_uncertainty:.4f}")
            print(f"Uncertainty-error correlation: {uncertainty_correlation:.4f}")

        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            training_time_per_epoch=np.mean(epoch_times),
            inference_time_per_sample=avg_inference_time,
            memory_usage_mb=memory_used,
            final_accuracy=final_accuracy,
            convergence_epoch=convergence_epoch,
            total_parameters=total_params,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
            uncertainty_available=uncertainty_available,
            mean_uncertainty=mean_uncertainty,
            uncertainty_correlation_with_errors=uncertainty_correlation
        )

        return result

    def run_benchmarks(self, n_epochs=5):
        """Run all benchmarks."""

        print("Loading data...")
        train_data, test_data = self.load_imdb_data(max_samples=self.max_samples)

        # Common config
        config = {
            'd_model': 256,  # Smaller for faster benchmarking
            'n_heads': 4,
            'vocab_size': 10000,
            'dropout': 0.3,
            'k_permutations': 10,
            'epsilon': 0.05
        }

        # Benchmark 1: Standard Transformer
        print("\n" + "="*60)
        print("BENCHMARK 1: Standard PyTorch Transformer")
        print("="*60)
        standard_model = StandardTransformer(config)
        self.results['Standard Transformer'] = self.train_and_evaluate(
            standard_model,
            'Standard Transformer',
            train_data,
            test_data,
            n_epochs=n_epochs,
            batch_size=8
        )

        # Benchmark 2: Bayesian Transformer
        print("\n" + "="*60)
        print("BENCHMARK 2: Bayesian Expectation Transformer")
        print("="*60)
        bayesian_model = BayesianTransformerWrapper(config)
        self.results['Bayesian Transformer'] = self.train_and_evaluate(
            bayesian_model,
            'Bayesian Transformer',
            train_data,
            test_data,
            n_epochs=n_epochs,
            batch_size=8
        )

    def generate_comparison_report(self) -> str:
        """Generate markdown comparison report."""

        report = ["# Transformer Benchmark Report\n"]
        report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Device**: {self.device}\n")
        report.append(f"**Dataset**: IMDB (max {self.max_samples} samples)\n\n")

        report.append("## Summary Comparison\n\n")
        report.append("| Metric | Standard Transformer | Bayesian Transformer | Winner |\n")
        report.append("|--------|----------------------|----------------------|--------|\n")

        std_result = self.results['Standard Transformer']
        bay_result = self.results['Bayesian Transformer']

        # Accuracy
        acc_winner = "Bayesian" if bay_result.final_accuracy > std_result.final_accuracy else "Standard"
        if abs(bay_result.final_accuracy - std_result.final_accuracy) < 0.01:
            acc_winner = "Tie"
        report.append(f"| **Accuracy** | {std_result.final_accuracy:.4f} | {bay_result.final_accuracy:.4f} | {acc_winner} |\n")

        # Training speed
        speed_winner = "Standard" if std_result.training_time_per_epoch < bay_result.training_time_per_epoch else "Bayesian"
        report.append(f"| **Training Speed** | {std_result.training_time_per_epoch:.2f}s/epoch | {bay_result.training_time_per_epoch:.2f}s/epoch | {speed_winner} |\n")

        # Inference speed
        inf_winner = "Standard" if std_result.inference_time_per_sample < bay_result.inference_time_per_sample else "Bayesian"
        report.append(f"| **Inference Speed** | {std_result.inference_time_per_sample:.2f}ms | {bay_result.inference_time_per_sample:.2f}ms | {inf_winner} |\n")

        # Memory
        mem_winner = "Standard" if std_result.memory_usage_mb < bay_result.memory_usage_mb else "Bayesian"
        report.append(f"| **Memory Usage** | {std_result.memory_usage_mb:.2f}MB | {bay_result.memory_usage_mb:.2f}MB | {mem_winner} |\n")

        # Parameters
        param_winner = "Standard" if std_result.total_parameters < bay_result.total_parameters else "Bayesian"
        report.append(f"| **Parameters** | {std_result.total_parameters:,} | {bay_result.total_parameters:,} | {param_winner} |\n")

        # Convergence
        conv_winner = "Standard" if std_result.convergence_epoch < bay_result.convergence_epoch else "Bayesian"
        report.append(f"| **Convergence (80% acc)** | Epoch {std_result.convergence_epoch} | Epoch {bay_result.convergence_epoch} | {conv_winner} |\n")

        # Uncertainty
        report.append(f"| **Uncertainty Quantification** | ❌ Not available | ✅ Available (mean: {bay_result.mean_uncertainty:.4f}) | Bayesian |\n")

        report.append("\n## Detailed Analysis\n\n")

        report.append("### Training Performance\n\n")
        report.append(f"- **Standard Transformer**: {std_result.training_time_per_epoch:.2f}s per epoch\n")
        report.append(f"- **Bayesian Transformer**: {bay_result.training_time_per_epoch:.2f}s per epoch\n")
        overhead = ((bay_result.training_time_per_epoch / std_result.training_time_per_epoch) - 1) * 100
        report.append(f"- **Overhead**: {overhead:.1f}% slower\n\n")

        report.append("### Inference Performance\n\n")
        report.append(f"- **Standard Transformer**: {std_result.inference_time_per_sample:.2f}ms per sample\n")
        report.append(f"- **Bayesian Transformer**: {bay_result.inference_time_per_sample:.2f}ms per sample\n")
        inf_overhead = ((bay_result.inference_time_per_sample / std_result.inference_time_per_sample) - 1) * 100
        report.append(f"- **Overhead**: {inf_overhead:.1f}% slower\n\n")

        report.append("### Uncertainty Quantification (Bayesian Only)\n\n")
        if bay_result.uncertainty_available:
            report.append(f"- **Mean Uncertainty**: {bay_result.mean_uncertainty:.4f}\n")
            report.append(f"- **Uncertainty-Error Correlation**: {bay_result.uncertainty_correlation_with_errors:.4f}\n")
            if bay_result.uncertainty_correlation_with_errors > 0.3:
                report.append("- ✅ **Good correlation**: Higher uncertainty on incorrect predictions\n\n")
            else:
                report.append("- ⚠️ **Weak correlation**: Uncertainty may need calibration\n\n")

        report.append("### Key Findings\n\n")

        if bay_result.final_accuracy > std_result.final_accuracy:
            report.append("✅ **Bayesian Transformer achieves higher accuracy**\n\n")

        if bay_result.uncertainty_correlation_with_errors > 0.3:
            report.append("✅ **Bayesian uncertainty is well-calibrated** (correlates with errors)\n\n")

        if overhead < 50:
            report.append("✅ **Reasonable overhead** (<50% slower than standard)\n\n")
        else:
            report.append("⚠️ **Significant overhead** (>50% slower than standard)\n\n")

        report.append("### Recommendations\n\n")

        if bay_result.uncertainty_available and bay_result.uncertainty_correlation_with_errors > 0.3:
            report.append("**Use Bayesian Transformer when:**\n")
            report.append("- Uncertainty quantification is critical (medical, finance, safety)\n")
            report.append("- Active learning is needed (identify uncertain samples)\n")
            report.append("- Calibrated confidence scores are required\n\n")

        report.append("**Use Standard Transformer when:**\n")
        report.append("- Speed is critical and uncertainty is not needed\n")
        report.append("- Memory is constrained\n")
        report.append("- Simple classification without confidence scores\n\n")

        return "".join(report)

    def visualize_results(self, output_dir='benchmarks/results'):
        """Create visualization plots."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        std_result = self.results['Standard Transformer']
        bay_result = self.results['Bayesian Transformer']

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Training curves
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss history
        axes[0, 0].plot(std_result.loss_history, label='Standard', marker='o')
        axes[0, 0].plot(bay_result.loss_history, label='Bayesian', marker='s')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy history
        axes[0, 1].plot(std_result.accuracy_history, label='Standard', marker='o')
        axes[0, 1].plot(bay_result.accuracy_history, label='Bayesian', marker='s')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training Accuracy Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Performance comparison (bar chart)
        metrics = ['Training\nSpeed\n(s/epoch)', 'Inference\nSpeed\n(ms)', 'Memory\n(MB)', 'Parameters\n(M)']
        std_values = [
            std_result.training_time_per_epoch,
            std_result.inference_time_per_sample,
            std_result.memory_usage_mb,
            std_result.total_parameters / 1e6
        ]
        bay_values = [
            bay_result.training_time_per_epoch,
            bay_result.inference_time_per_sample,
            bay_result.memory_usage_mb,
            bay_result.total_parameters / 1e6
        ]

        x = np.arange(len(metrics))
        width = 0.35

        axes[1, 0].bar(x - width/2, std_values, width, label='Standard', alpha=0.8)
        axes[1, 0].bar(x + width/2, bay_values, width, label='Bayesian', alpha=0.8)
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Performance Metrics Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(metrics)
        axes[1, 0].legend()
        axes[1, 0].grid(True, axis='y')

        # Accuracy comparison (bar chart)
        models = ['Standard\nTransformer', 'Bayesian\nTransformer']
        accuracies = [std_result.final_accuracy * 100, bay_result.final_accuracy * 100]
        colors = ['#3498db', '#e74c3c']

        bars = axes[1, 1].bar(models, accuracies, color=colors, alpha=0.8)
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Final Test Accuracy')
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].grid(True, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path / 'benchmark_comparison.png'}")
        plt.close()

        # Plot 2: Detailed speedup analysis
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Training', 'Inference']
        speedup = [
            std_result.training_time_per_epoch / bay_result.training_time_per_epoch,
            std_result.inference_time_per_sample / bay_result.inference_time_per_sample
        ]

        colors_speedup = ['green' if s > 1 else 'red' for s in speedup]
        bars = ax.barh(categories, speedup, color=colors_speedup, alpha=0.7)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='Baseline (1x)')
        ax.set_xlabel('Speedup Factor (>1 = Bayesian faster)')
        ax.set_title('Speed Comparison: Standard vs Bayesian')
        ax.legend()
        ax.grid(True, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, speedup)):
            label = f'{val:.2f}x'
            if val > 1:
                label += ' faster'
            else:
                label += ' slower'
            ax.text(val, bar.get_y() + bar.get_height()/2, label,
                   va='center', ha='left' if val < 1 else 'right', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path / 'speedup_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path / 'speedup_analysis.png'}")
        plt.close()

    def save_results(self, output_dir='benchmarks/results'):
        """Save benchmark results to JSON."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert results to dict, handling numpy types
        results_dict = {}
        for name, result in self.results.items():
            result_dict = asdict(result)
            # Convert numpy/bool types to native Python
            for key, value in result_dict.items():
                if isinstance(value, (np.bool_, bool)):
                    result_dict[key] = bool(value)
                elif isinstance(value, (np.integer, np.floating)):
                    result_dict[key] = float(value) if '.' in str(value) else int(value)
                elif isinstance(value, list):
                    result_dict[key] = [float(x) if isinstance(x, (np.floating, float)) else x for x in value]
            results_dict[name] = result_dict

        # Save to JSON
        json_path = output_path / 'benchmark_results.json'
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"Saved results: {json_path}")

        # Save markdown report
        report = self.generate_comparison_report()
        md_path = output_path / 'BENCHMARK_REPORT.md'
        with open(md_path, 'w') as f:
            f.write(report)

        print(f"Saved report: {md_path}")


def main():
    """Run complete benchmark suite."""

    print("="*60)
    print("TRANSFORMER BENCHMARK SUITE")
    print("="*60)
    print()

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Initialize benchmark
    benchmark = TransformerBenchmark(device=device, max_samples=1000)

    # Run benchmarks
    benchmark.run_benchmarks(n_epochs=5)

    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)

    report = benchmark.generate_comparison_report()
    print(report)

    # Save results
    benchmark.save_results()

    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    benchmark.visualize_results()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)
    print("\nResults saved to: benchmarks/results/")
    print("- benchmark_results.json")
    print("- BENCHMARK_REPORT.md")
    print("- benchmark_comparison.png")
    print("- speedup_analysis.png")


if __name__ == '__main__':
    main()
