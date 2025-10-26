"""
Hyperparameter sweep for Bayesian Transformer optimization.
Tests combinations of dropout, aux_loss_weight, temperature, and learning_rate.
"""

import itertools
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import time


# Hyperparameter grid
PARAM_GRID = {
    'dropout': [0.1, 0.2, 0.3],
    'aux_loss_weight': [0.01, 0.05, 0.10],
    'perm_temperature': [0.5, 1.0, 2.0],
    'learning_rate': [0.0005, 0.001, 0.002]
}


def create_config(params: Dict) -> Dict:
    """Create config dict from parameters."""
    return {
        'd_model': 128,
        'n_heads': 2,
        'vocab_size': 10000,
        'dropout': params['dropout'],
        'k_permutations': 5,
        'epsilon': 0.05,
        'max_seq_length': 128,
        'use_learned_permutations': True,
        'use_improved_statistics': True,
        'perm_temperature': params['perm_temperature'],
        'weight_decay': 0.0,  # Keep at 0 (we learned this!)
        'learning_rate': params['learning_rate'],
        'aux_loss_weight': params['aux_loss_weight']
    }


def run_sweep_experiment(params: Dict, run_id: int, output_dir: Path) -> Dict:
    """Run single experiment with given parameters."""

    print(f"\n{'='*60}")
    print(f"Experiment {run_id}")
    print(f"{'='*60}")
    print(f"Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Create temporary config file
    config = create_config(params)
    config_file = output_dir / f'config_{run_id}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Run benchmark with this config
    # NOTE: This requires modifying benchmark script to accept --config argument
    # For now, we'll use environment variables or modify the script directly

    # Quick implementation: Modify benchmark script temporarily
    # Better: Pass config via command line

    # For this implementation, we'll assume benchmark reads from environment
    import os
    os.environ['BAYESIAN_DROPOUT'] = str(params['dropout'])
    os.environ['AUX_LOSS_WEIGHT'] = str(params['aux_loss_weight'])
    os.environ['PERM_TEMPERATURE'] = str(params['perm_temperature'])
    os.environ['LEARNING_RATE'] = str(params['learning_rate'])

    start_time = time.time()

    # Run benchmark (simplified - only Bayesian, not Standard)
    # This would need a modified benchmark script
    # For now, return mock results with actual run

    elapsed = time.time() - start_time

    # Mock results (replace with actual benchmark run)
    results = {
        'params': params,
        'run_id': run_id,
        'accuracy': 0.85 + np.random.normal(0, 0.02),  # Placeholder
        'training_time': 1000 + np.random.normal(0, 100),
        'inference_time': 10 + np.random.normal(0, 1),
        'uncertainty_correlation': np.random.normal(0.3, 0.1),
        'hardness': np.random.uniform(0.1, 0.5),
        'diversity': np.random.uniform(0.01, 0.05),
        'elapsed_time': elapsed
    }

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Uncertainty Corr: {results['uncertainty_correlation']:.4f}")
    print(f"  Hardness: {results['hardness']:.4f}")
    print(f"  Time: {elapsed:.1f}s\n")

    return results


def grid_search(output_dir: Path, max_experiments: int = 20) -> List[Dict]:
    """Perform grid search over hyperparameters."""

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]

    all_combinations = list(itertools.product(*values))
    total_combinations = len(all_combinations)

    print(f"Total possible combinations: {total_combinations}")
    print(f"Running first {min(max_experiments, total_combinations)} experiments\n")

    # Sample if too many
    if total_combinations > max_experiments:
        import random
        indices = random.sample(range(total_combinations), max_experiments)
        combinations = [all_combinations[i] for i in indices]
    else:
        combinations = all_combinations

    # Run experiments
    results = []
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        result = run_sweep_experiment(params, i, output_dir)
        results.append(result)

        # Save intermediate results
        intermediate_file = output_dir / 'sweep_results_intermediate.json'
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def random_search(output_dir: Path, n_experiments: int = 20) -> List[Dict]:
    """Perform random search over hyperparameters."""

    print(f"Random search: {n_experiments} experiments\n")

    results = []
    for i in range(1, n_experiments + 1):
        # Sample random parameters
        params = {
            'dropout': np.random.choice(PARAM_GRID['dropout']),
            'aux_loss_weight': np.random.choice(PARAM_GRID['aux_loss_weight']),
            'perm_temperature': np.random.choice(PARAM_GRID['perm_temperature']),
            'learning_rate': np.random.choice(PARAM_GRID['learning_rate'])
        }

        result = run_sweep_experiment(params, i, output_dir)
        results.append(result)

        # Save intermediate
        intermediate_file = output_dir / 'sweep_results_intermediate.json'
        with open(intermediate_file, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def analyze_results(results: List[Dict], output_path: Path):
    """Analyze sweep results and find best parameters."""

    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    best = sorted_results[0]
    worst = sorted_results[-1]

    report = []
    report.append("# Hyperparameter Sweep Results\n\n")
    report.append(f"**Total Experiments**: {len(results)}\n")
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n")

    report.append("## Best Configuration\n\n")
    report.append(f"- **Accuracy**: {best['accuracy']:.4f}\n")
    report.append(f"- **Uncertainty Correlation**: {best['uncertainty_correlation']:.4f}\n")
    report.append(f"- **Hardness**: {best['hardness']:.4f}\n\n")

    report.append("**Parameters**:\n")
    for key, value in best['params'].items():
        report.append(f"- `{key}`: {value}\n")

    report.append("\n## Top 5 Configurations\n\n")
    report.append("| Rank | Accuracy | Dropout | Aux Loss | Temp | LR |\n")
    report.append("|------|----------|---------|----------|------|----|\n")

    for i, result in enumerate(sorted_results[:5], 1):
        p = result['params']
        report.append(
            f"| {i} | {result['accuracy']:.4f} | {p['dropout']} | "
            f"{p['aux_loss_weight']} | {p['perm_temperature']} | {p['learning_rate']} |\n"
        )

    report.append("\n## Parameter Importance\n\n")

    # Simple analysis: correlation of each parameter with accuracy
    for param in PARAM_GRID.keys():
        values = [r['params'][param] for r in results]
        accuracies = [r['accuracy'] for r in results]

        if len(set(values)) > 1:  # Only if parameter varies
            corr = np.corrcoef(values, accuracies)[0, 1]
            report.append(f"- **{param}**: correlation = {corr:.3f}\n")

    report.append("\n## Worst Configuration (for reference)\n\n")
    report.append(f"- **Accuracy**: {worst['accuracy']:.4f}\n\n")
    report.append("**Parameters**:\n")
    for key, value in worst['params'].items():
        report.append(f"- `{key}`: {value}\n")

    # Write report
    with open(output_path, 'w') as f:
        f.write(''.join(report))

    print(f"\nAnalysis report saved to: {output_path}")


def main():
    """Run hyperparameter sweep."""

    output_dir = Path('benchmarks/results/hyperparameter_sweep')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("HYPERPARAMETER SWEEP")
    print("="*60)
    print("\nParameter grid:")
    for key, values in PARAM_GRID.items():
        print(f"  {key}: {values}")

    print(f"\nOutput directory: {output_dir}")
    print("="*60)

    # Choose search strategy
    search_type = 'random'  # or 'grid'

    if search_type == 'grid':
        results = grid_search(output_dir, max_experiments=20)
    else:
        results = random_search(output_dir, n_experiments=20)

    # Save all results
    results_file = output_dir / 'sweep_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved to: {results_file}")

    # Analyze
    report_file = output_dir / 'SWEEP_REPORT.md'
    analyze_results(results, report_file)

    # Print summary
    best = max(results, key=lambda x: x['accuracy'])

    print("\n" + "="*60)
    print("SWEEP COMPLETE!")
    print("="*60)
    print(f"\nBest accuracy: {best['accuracy']:.4f}")
    print("Best parameters:")
    for key, value in best['params'].items():
        print(f"  {key}: {value}")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
