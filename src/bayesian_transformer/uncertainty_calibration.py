"""
Uncertainty calibration module for Bayesian Transformer.

Implements:
1. Temperature Scaling - Post-hoc calibration via learned temperature
2. Platt Scaling - Logistic regression for calibration
3. Isotonic Regression - Non-parametric calibration
4. ECE/MCE computation - Calibration metrics

Goal: Improve uncertainty-error correlation from 0.0 to >0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for uncertainty calibration.

    Learns a single scalar temperature parameter T that scales uncertainties:
    calibrated_uncertainty = uncertainty / T

    Optimized on validation set to minimize negative log-likelihood.
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling."""
        # Clamp temperature to reasonable range
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        return uncertainty / temp

    def calibrate(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Learn optimal temperature on validation set.

        Args:
            uncertainties: Model uncertainty estimates (N,)
            errors: Actual errors (N,) - typically binary (0/1) for classification
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            calibrated = self.forward(uncertainties)

            # Negative log-likelihood: high uncertainty should predict high error
            # Use MSE as proxy (could use actual NLL if we have probabilities)
            loss = F.mse_loss(calibrated, errors)

            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"Temperature Scaling: Learned T = {self.temperature.item():.4f}")


class PlattScaling:
    """
    Platt scaling for uncertainty calibration.

    Fits logistic regression: P(error=1) = sigmoid(A * uncertainty + B)
    More flexible than temperature scaling (two parameters instead of one).
    """

    def __init__(self):
        self.model = LogisticRegression()
        self.fitted = False

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray):
        """
        Learn Platt scaling parameters.

        Args:
            uncertainties: Model uncertainty estimates (N,)
            errors: Actual errors (N,) - binary (0/1)
        """
        # Reshape for sklearn
        X = uncertainties.reshape(-1, 1)
        y = errors.ravel()

        self.model.fit(X, y)
        self.fitted = True

        print(f"Platt Scaling: A = {self.model.coef_[0][0]:.4f}, B = {self.model.intercept_[0]:.4f}")

    def transform(self, uncertainties: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        X = uncertainties.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]  # Probability of error


class IsotonicCalibration:
    """
    Isotonic regression for uncertainty calibration.

    Non-parametric method that learns monotonic mapping from
    uncertainty to calibrated probability.

    Most flexible but requires more data to avoid overfitting.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray):
        """
        Learn isotonic mapping.

        Args:
            uncertainties: Model uncertainty estimates (N,)
            errors: Actual errors (N,) - binary (0/1)
        """
        self.model.fit(uncertainties.ravel(), errors.ravel())
        self.fitted = True

        print("Isotonic Calibration: Fitted non-parametric mapping")

    def transform(self, uncertainties: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")

        return self.model.transform(uncertainties.ravel())


def compute_calibration_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE: Average absolute difference between predicted uncertainty and actual error rate
    MCE: Maximum absolute difference across all bins

    Args:
        uncertainties: Model uncertainty estimates (N,)
        errors: Actual errors (N,) - binary (0/1)
        n_bins: Number of bins for calibration plot

    Returns:
        Dict with ECE, MCE, and per-bin statistics
    """
    uncertainties = uncertainties.ravel()
    errors = errors.ravel()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Accuracy in bin (1 - error_rate for this bin)
            accuracy_in_bin = 1.0 - errors[in_bin].mean()

            # Average confidence in bin (use uncertainty as confidence proxy)
            avg_confidence_in_bin = uncertainties[in_bin].mean()

            # Calibration error for this bin
            bin_error = abs(avg_confidence_in_bin - (1 - accuracy_in_bin))

            ece += bin_error * prop_in_bin
            mce = max(mce, bin_error)

            bin_stats.append({
                'bin': f'({bin_lower:.2f}, {bin_upper:.2f}]',
                'count': in_bin.sum(),
                'avg_uncertainty': avg_confidence_in_bin,
                'error_rate': errors[in_bin].mean(),
                'calibration_error': bin_error
            })

    return {
        'ECE': ece,
        'MCE': mce,
        'bin_stats': bin_stats
    }


def calibrate_uncertainties(
    train_uncertainties: torch.Tensor,
    train_errors: torch.Tensor,
    val_uncertainties: torch.Tensor,
    val_errors: torch.Tensor,
    method: str = 'temperature'
) -> Tuple[callable, Dict]:
    """
    Calibrate uncertainties using specified method.

    Args:
        train_uncertainties: Training set uncertainties
        train_errors: Training set errors (binary)
        val_uncertainties: Validation set uncertainties
        val_errors: Validation set errors (binary)
        method: 'temperature', 'platt', or 'isotonic'

    Returns:
        calibration_function: Function to apply to new uncertainties
        metrics: Calibration metrics before and after
    """
    # Convert to numpy for sklearn
    train_unc_np = train_uncertainties.cpu().numpy()
    train_err_np = train_errors.cpu().numpy()
    val_unc_np = val_uncertainties.cpu().numpy()
    val_err_np = val_errors.cpu().numpy()

    # Compute metrics before calibration
    metrics_before = compute_calibration_metrics(val_unc_np, val_err_np)

    # Fit calibration model
    if method == 'temperature':
        calibrator = TemperatureScaling()
        calibrator.calibrate(train_uncertainties, train_errors)

        def calibrate_fn(unc):
            if isinstance(unc, np.ndarray):
                unc = torch.from_numpy(unc).float()
            return calibrator(unc).detach().cpu().numpy()

    elif method == 'platt':
        calibrator = PlattScaling()
        calibrator.fit(train_unc_np, train_err_np)
        calibrate_fn = calibrator.transform

    elif method == 'isotonic':
        calibrator = IsotonicCalibration()
        calibrator.fit(train_unc_np, train_err_np)
        calibrate_fn = calibrator.transform

    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Apply to validation set
    calibrated_val = calibrate_fn(val_unc_np)

    # Compute metrics after calibration
    metrics_after = compute_calibration_metrics(calibrated_val, val_err_np)

    # Compute correlation improvement
    corr_before = np.corrcoef(val_unc_np.ravel(), val_err_np.ravel())[0, 1]
    corr_after = np.corrcoef(calibrated_val.ravel(), val_err_np.ravel())[0, 1]

    metrics = {
        'before': {
            'ECE': metrics_before['ECE'],
            'MCE': metrics_before['MCE'],
            'correlation': corr_before
        },
        'after': {
            'ECE': metrics_after['ECE'],
            'MCE': metrics_after['MCE'],
            'correlation': corr_after
        },
        'improvement': {
            'ECE': metrics_before['ECE'] - metrics_after['ECE'],
            'MCE': metrics_before['MCE'] - metrics_after['MCE'],
            'correlation': corr_after - corr_before
        }
    }

    print("\n" + "="*60)
    print(f"CALIBRATION RESULTS ({method.upper()})")
    print("="*60)
    print(f"ECE:         {metrics['before']['ECE']:.4f} -> {metrics['after']['ECE']:.4f} (delta {metrics['improvement']['ECE']:.4f})")
    print(f"MCE:         {metrics['before']['MCE']:.4f} -> {metrics['after']['MCE']:.4f} (delta {metrics['improvement']['MCE']:.4f})")
    print(f"Correlation: {metrics['before']['correlation']:.4f} -> {metrics['after']['correlation']:.4f} (delta {metrics['improvement']['correlation']:.4f})")
    print("="*60 + "\n")

    return calibrate_fn, metrics


def compare_calibration_methods(
    train_uncertainties: torch.Tensor,
    train_errors: torch.Tensor,
    val_uncertainties: torch.Tensor,
    val_errors: torch.Tensor
) -> Dict:
    """
    Compare all calibration methods and return best.

    Returns:
        Dict with results for each method and best method recommendation
    """
    methods = ['temperature', 'platt', 'isotonic']
    results = {}

    print("\n" + "="*60)
    print("COMPARING CALIBRATION METHODS")
    print("="*60 + "\n")

    for method in methods:
        calibrate_fn, metrics = calibrate_uncertainties(
            train_uncertainties, train_errors,
            val_uncertainties, val_errors,
            method=method
        )
        results[method] = {
            'calibrate_fn': calibrate_fn,
            'metrics': metrics
        }

    # Find best method (lowest ECE, highest correlation)
    best_method = min(
        methods,
        key=lambda m: results[m]['metrics']['after']['ECE'] - results[m]['metrics']['after']['correlation']
    )

    print("\n" + "="*60)
    print(f"BEST METHOD: {best_method.upper()}")
    print("="*60)
    print(f"Final ECE:         {results[best_method]['metrics']['after']['ECE']:.4f}")
    print(f"Final Correlation: {results[best_method]['metrics']['after']['correlation']:.4f}")
    print("="*60 + "\n")

    results['best_method'] = best_method
    results['best_calibrate_fn'] = results[best_method]['calibrate_fn']

    return results
