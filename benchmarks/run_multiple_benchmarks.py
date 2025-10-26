"""
Run multiple benchmark iterations for statistical significance.
Computes mean, std, confidence intervals for all metrics.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import time


def run_single_benchmark(seed: int, output_dir: Path) -> Dict:
    """Run benchmark with specific seed."""
    print(f"\n{'='*60}")
    print(f"Running benchmark with seed {seed}")
    print(f"{'='*60}\n")

    # Set environment variable for reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Run benchmark
    cmd = [
        'python',
        'benchmarks/benchmark_transformer_comparison.py',
        '--seed', str(seed),
        '--output-dir', str(output_dir / f'run_{seed}')
    ]

    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path.cwd()
    )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR in run {seed}:")
        print(result.stderr)
        return None

    # Load results
    results_file = output_dir / f'run_{seed}' / 'benchmark_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)

        print(f"Run {seed} completed in {elapsed:.1f}s")
        print(f"  Standard: {results['Standard Transformer']['final_accuracy']:.4f}")
        print(f"  Bayesian: {results['Bayesian Transformer']['final_accuracy']:.4f}")

        return results
    else:
        print(f"ERROR: Results file not found for seed {seed}")
        return None


def compute_statistics(all_results: List[Dict]) -> Dict:
    """Compute mean, std, CI for all metrics."""

    # Extract metrics for each model
    models = ['Standard Transformer', 'Bayesian Transformer']
    stats = {}

    for model in models:
        accuracies = [r[model]['final_accuracy'] for r in all_results]
        train_times = [r[model]['training_time_per_epoch'] for r in all_results]
        infer_times = [r[model]['inference_time_per_sample'] for r in all_results]

        # Compute statistics
        stats[model] = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies, ddof=1),
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'ci_95': 1.96 * np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))
            },
            'training_time': {
                'mean': np.mean(train_times),
                'std': np.std(train_times, ddof=1)
            },
            'inference_time': {
                'mean': np.mean(infer_times),
                'std': np.std(infer_times, ddof=1)
            }
        }

        # Uncertainty metrics (if available)
        if all_results[0][model].get('uncertainty_available', False):
            uncertainties = [r[model]['mean_uncertainty'] for r in all_results]
            correlations = [r[model]['uncertainty_correlation_with_errors'] for r in all_results]

            stats[model]['uncertainty'] = {
                'mean': np.mean(uncertainties),
                'std': np.std(uncertainties, ddof=1)
            }
            stats[model]['correlation'] = {
                'mean': np.mean(correlations),
                'std': np.std(correlations, ddof=1),
                'ci_95': 1.96 * np.std(correlations, ddof=1) / np.sqrt(len(correlations))
            }

    return stats


def t_test(stats: Dict, alpha: float = 0.05):
    """Perform t-test between Standard and Bayesian."""
    from scipy import stats as scipy_stats

    std_acc = stats['Standard Transformer']['accuracy']
    bay_acc = stats['Bayesian Transformer']['accuracy']

    # Two-sample t-test
    # Assuming equal variance (can test this separately)
    n = 5  # Number of runs

    # Pooled standard deviation
    pooled_std = np.sqrt((std_acc['std']**2 + bay_acc['std']**2) / 2)

    # T-statistic
    t_stat = (std_acc['mean'] - bay_acc['mean']) / (pooled_std * np.sqrt(2/n))

    # Degrees of freedom
    df = 2 * n - 2

    # P-value (two-tailed)
    from scipy.stats import t as t_dist
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))

    significant = p_value < alpha

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'significant': significant,
        'alpha': alpha,
        'interpretation': 'Significant difference' if significant else 'No significant difference'
    }


def generate_report(stats: Dict, t_test_result: Dict, output_path: Path):
    """Generate markdown report."""

    report = []
    report.append("# Multiple Runs Statistical Analysis\n")
    report.append(f"**Number of Runs**: 5\n")
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")

    report.append("## Summary Statistics\n\n")

    # Table
    report.append("| Model | Accuracy (Mean ± Std) | 95% CI | Min-Max |\n")
    report.append("|-------|----------------------|---------|----------|\n")

    for model in ['Standard Transformer', 'Bayesian Transformer']:
        acc = stats[model]['accuracy']
        report.append(
            f"| {model} | {acc['mean']:.4f} ± {acc['std']:.4f} | "
            f"±{acc['ci_95']:.4f} | {acc['min']:.4f}-{acc['max']:.4f} |\n"
        )

    report.append("\n## T-Test Results\n\n")
    report.append(f"- **T-statistic**: {t_test_result['t_statistic']:.4f}\n")
    report.append(f"- **P-value**: {t_test_result['p_value']:.4f}\n")
    report.append(f"- **Degrees of Freedom**: {t_test_result['degrees_of_freedom']}\n")
    report.append(f"- **Significance Level**: α = {t_test_result['alpha']}\n")
    report.append(f"- **Result**: **{t_test_result['interpretation']}**\n\n")

    if t_test_result['significant']:
        report.append("The difference in accuracy is **statistically significant**.\n\n")
    else:
        report.append("The difference in accuracy is **not statistically significant**.\n")
        report.append("Bayesian and Standard performance is statistically equivalent.\n\n")

    # Uncertainty (if available)
    if 'correlation' in stats['Bayesian Transformer']:
        corr = stats['Bayesian Transformer']['correlation']
        report.append("## Uncertainty Correlation\n\n")
        report.append(f"- **Mean Correlation**: {corr['mean']:.4f} ± {corr['std']:.4f}\n")
        report.append(f"- **95% CI**: ±{corr['ci_95']:.4f}\n\n")

        if corr['mean'] > 0.3:
            report.append("Uncertainty shows **good positive correlation** with errors!\n\n")
        elif corr['mean'] > 0:
            report.append("Uncertainty shows **weak positive correlation** with errors.\n\n")
        else:
            report.append("Uncertainty still has **negative or no correlation** with errors.\n\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    print(f"\nReport saved to: {output_path}")


def main():
    """Run multiple benchmarks and compute statistics."""

    # Configuration
    n_runs = 5
    seeds = [42, 123, 456, 789, 1024]  # Fixed seeds for reproducibility
    output_dir = Path('benchmarks/results/multiple_runs')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MULTIPLE BENCHMARK RUNS FOR STATISTICAL SIGNIFICANCE")
    print("="*60)
    print(f"Number of runs: {n_runs}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    # Run benchmarks
    all_results = []
    for seed in seeds:
        result = run_single_benchmark(seed, output_dir)
        if result is not None:
            all_results.append(result)
        else:
            print(f"WARNING: Run with seed {seed} failed, skipping...")

    if len(all_results) < 2:
        print("ERROR: Not enough successful runs for statistics")
        return

    print(f"\n{'='*60}")
    print(f"COMPUTING STATISTICS ({len(all_results)} successful runs)")
    print(f"{'='*60}\n")

    # Compute statistics
    stats = compute_statistics(all_results)

    # T-test
    try:
        t_result = t_test(stats)
    except Exception as e:
        print(f"WARNING: Could not perform t-test: {e}")
        t_result = {'interpretation': 'Could not compute'}

    # Save statistics
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump({
            'statistics': stats,
            't_test': t_result,
            'n_runs': len(all_results),
            'seeds': seeds[:len(all_results)]
        }, f, indent=2)

    print(f"Statistics saved to: {stats_file}")

    # Generate report
    report_file = output_dir / 'STATISTICAL_REPORT.md'
    generate_report(stats, t_result, report_file)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    std_acc = stats['Standard Transformer']['accuracy']
    bay_acc = stats['Bayesian Transformer']['accuracy']

    print(f"\nStandard Transformer:")
    print(f"  Accuracy: {std_acc['mean']:.4f} ± {std_acc['std']:.4f}")
    print(f"  95% CI: [{std_acc['mean']-std_acc['ci_95']:.4f}, {std_acc['mean']+std_acc['ci_95']:.4f}]")

    print(f"\nBayesian Transformer:")
    print(f"  Accuracy: {bay_acc['mean']:.4f} ± {bay_acc['std']:.4f}")
    print(f"  95% CI: [{bay_acc['mean']-bay_acc['ci_95']:.4f}, {bay_acc['mean']+bay_acc['ci_95']:.4f}]")

    print(f"\nDifference: {std_acc['mean'] - bay_acc['mean']:.4f}")
    print(f"T-test: {t_result['interpretation']}")

    print("\n" + "="*60)
    print("ALL RUNS COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    # Install scipy if needed
    try:
        import scipy
    except ImportError:
        print("Installing scipy...")
        subprocess.run(['pip', 'install', 'scipy'])

    main()
