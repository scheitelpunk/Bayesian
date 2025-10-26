"""
Uncertainty Calibration for Bayesian Transformer

Implements temperature scaling and Platt scaling for calibrating
epistemic and aleatoric uncertainty estimates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple


class UncertaintyCalibrator(nn.Module):
    """
    Temperature scaling for uncertainty calibration.

    Learns a temperature parameter that scales uncertainty estimates
    to better correlate with actual prediction errors.
    """

    def __init__(self, initial_temperature: float = 1.5):
        """
        Initialize calibrator with temperature parameter.

        Args:
            initial_temperature: Starting temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to uncertainty.

        Args:
            uncertainty: Raw uncertainty estimates (batch_size,)

        Returns:
            Calibrated uncertainty (batch_size,)
        """
        return uncertainty / torch.clamp(self.temperature, min=0.1, max=10.0)

    def fit(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False
    ):
        """
        Fit temperature parameter using validation set.

        Args:
            uncertainties: Raw uncertainty estimates (n_samples,)
            errors: Prediction errors (0 = correct, 1 = wrong) (n_samples,)
            lr: Learning rate
            max_iter: Maximum optimization iterations
            verbose: Print optimization progress
        """
        uncertainties = uncertainties.detach()
        errors = errors.float()

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()

            # Calibrated uncertainties
            calibrated = self(uncertainties)

            # Loss: MSE between calibrated uncertainty and errors
            # Want high uncertainty when error=1, low when error=0
            loss = nn.functional.mse_loss(calibrated, errors)

            loss.backward()

            if verbose:
                print(f"Temperature: {self.temperature.item():.4f}, Loss: {loss.item():.4f}")

            return loss

        optimizer.step(closure)

        if verbose:
            print(f"\nFinal temperature: {self.temperature.item():.4f}")


class PlattScaling:
    """
    Platt scaling for binary uncertainty calibration.

    Fits a logistic regression model to map uncertainty to error probability.
    """

    def __init__(self):
        self.a = None
        self.b = None

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray):
        """
        Fit Platt scaling parameters.

        Args:
            uncertainties: Raw uncertainty estimates (n_samples,)
            errors: Prediction errors (0 = correct, 1 = wrong) (n_samples,)
        """
        from sklearn.linear_model import LogisticRegression

        # Reshape for sklearn
        X = uncertainties.reshape(-1, 1)
        y = errors

        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(X, y)

        self.a = lr.coef_[0][0]
        self.b = lr.intercept_[0]

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to uncertainties.

        Args:
            uncertainties: Raw uncertainty estimates (n_samples,)

        Returns:
            Calibrated error probabilities (n_samples,)
        """
        if self.a is None or self.b is None:
            raise ValueError("Platt scaling not fitted. Call fit() first.")

        # Logistic function
        z = self.a * uncertainties + self.b
        calibrated = 1.0 / (1.0 + np.exp(-z))

        return calibrated


class IsotonicCalibration:
    """
    Isotonic regression for uncertainty calibration.

    Non-parametric calibration method that preserves monotonicity.
    """

    def __init__(self):
        self.calibrator = None

    def fit(self, uncertainties: np.ndarray, errors: np.ndarray):
        """
        Fit isotonic regression.

        Args:
            uncertainties: Raw uncertainty estimates (n_samples,)
            errors: Prediction errors (0 = correct, 1 = wrong) (n_samples,)
        """
        from sklearn.isotonic import IsotonicRegression

        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(uncertainties, errors)

    def calibrate(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            uncertainties: Raw uncertainty estimates (n_samples,)

        Returns:
            Calibrated error probabilities (n_samples,)
        """
        if self.calibrator is None:
            raise ValueError("Isotonic calibration not fitted. Call fit() first.")

        return self.calibrator.predict(uncertainties)


def compute_calibration_metrics(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute calibration metrics: ECE and MCE.

    Args:
        uncertainties: Uncertainty estimates (n_samples,)
        errors: Prediction errors (0 = correct, 1 = wrong) (n_samples,)
        n_bins: Number of bins for calibration curve

    Returns:
        Tuple of (ECE, MCE, bin_accuracies, bin_uncertainties)
    """
    # Sort by uncertainty
    sorted_indices = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_indices]
    sorted_errors = errors[sorted_indices]

    # Create bins
    n_samples = len(uncertainties)
    bin_size = n_samples // n_bins

    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    bin_accuracies = []
    bin_uncertainties = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples

        bin_errors = sorted_errors[start_idx:end_idx]
        bin_uncs = sorted_uncertainties[start_idx:end_idx]

        if len(bin_errors) == 0:
            continue

        # Average error rate in bin
        bin_error_rate = np.mean(bin_errors)

        # Average uncertainty in bin
        bin_avg_uncertainty = np.mean(bin_uncs)

        # Calibration error
        calibration_error = abs(bin_error_rate - bin_avg_uncertainty)

        # Weight by bin size
        bin_weight = len(bin_errors) / n_samples

        ece += bin_weight * calibration_error
        mce = max(mce, calibration_error)

        bin_accuracies.append(bin_error_rate)
        bin_uncertainties.append(bin_avg_uncertainty)

    return ece, mce, np.array(bin_accuracies), np.array(bin_uncertainties)


def plot_calibration_curve(
    uncertainties: np.ndarray,
    errors: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram (calibration curve).

    Args:
        uncertainties: Uncertainty estimates (n_samples,)
        errors: Prediction errors (0 = correct, 1 = wrong) (n_samples,)
        n_bins: Number of bins
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    ece, mce, bin_accuracies, bin_uncertainties = compute_calibration_metrics(
        uncertainties, errors, n_bins
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

    # Calibration curve
    ax.plot(bin_uncertainties, bin_accuracies, 'o-',
            label=f'Model (ECE={ece:.3f}, MCE={mce:.3f})',
            linewidth=2, markersize=8)

    ax.set_xlabel('Mean Predicted Uncertainty', fontsize=14)
    ax.set_ylabel('Fraction of Errors', fontsize=14)
    ax.set_title('Reliability Diagram', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")

    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    # Raw uncertainties (not calibrated)
    raw_uncertainties = np.random.beta(2, 5, n_samples)

    # Errors: higher uncertainty should mean more errors
    error_prob = raw_uncertainties * 0.8 + 0.1
    errors = (np.random.random(n_samples) < error_prob).astype(float)

    print("Raw Uncertainty Statistics:")
    print(f"  Mean: {raw_uncertainties.mean():.4f}")
    print(f"  Std: {raw_uncertainties.std():.4f}")
    print(f"  Correlation with errors: {np.corrcoef(raw_uncertainties, errors)[0,1]:.4f}")

    # Compute calibration metrics
    ece, mce, _, _ = compute_calibration_metrics(raw_uncertainties, errors)
    print(f"\nRaw Calibration Metrics:")
    print(f"  ECE: {ece:.4f}")
    print(f"  MCE: {mce:.4f}")

    # Temperature scaling
    print("\nTemperature Scaling:")
    calibrator = UncertaintyCalibrator(initial_temperature=2.0)
    calibrator.fit(
        torch.from_numpy(raw_uncertainties),
        torch.from_numpy(errors),
        verbose=True
    )

    calibrated_uncertainties = calibrator(
        torch.from_numpy(raw_uncertainties)
    ).detach().numpy()

    print(f"\nCalibrated Uncertainty Statistics:")
    print(f"  Mean: {calibrated_uncertainties.mean():.4f}")
    print(f"  Std: {calibrated_uncertainties.std():.4f}")
    print(f"  Correlation with errors: {np.corrcoef(calibrated_uncertainties, errors)[0,1]:.4f}")

    # Compute calibration metrics after calibration
    ece_cal, mce_cal, _, _ = compute_calibration_metrics(calibrated_uncertainties, errors)
    print(f"\nCalibrated Metrics:")
    print(f"  ECE: {ece_cal:.4f}")
    print(f"  MCE: {mce_cal:.4f}")

    # Plot calibration curve
    plot_calibration_curve(calibrated_uncertainties, errors, save_path='calibration_curve.png')
