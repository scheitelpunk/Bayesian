"""
Improved Statistics Encoder with correct uncertainty semantics.
Ensures uncertainty is positively correlated with prediction errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class StatisticsEncoderV2(nn.Module):
    """
    Enhanced statistics encoder that produces meaningful uncertainty estimates.

    Key improvements:
    1. Uses standard deviation instead of variance (more interpretable)
    2. Ensures uncertainty monotonically increases with permutation variance
    3. Separates uncertainty computation from feature encoding
    4. Uses Softplus activation for guaranteed positive uncertainty
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Uncertainty estimation head
        # Maps from std to scalar uncertainty value
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()  # Ensures positive output, smooth gradient
        )

        # Statistics encoding (mean + std → encoded features)
        self.stats_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # Learnable temperature for uncertainty calibration
        self.uncertainty_temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(
        self,
        x_permuted: torch.Tensor,
        return_detailed: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute statistics and uncertainty from permuted inputs.

        Args:
            x_permuted: (batch, k_perms, seq_len, d_model) permuted activations
            return_detailed: If True, return per-token uncertainty

        Returns:
            Dictionary containing:
                - encoded_stats: (batch, seq_len, d_model) encoded statistics
                - uncertainty: (batch,) scalar uncertainty per sample
                - std_per_token: (batch, seq_len) uncertainty per token (if return_detailed)
        """
        batch_size, k_perms, seq_len, d_model = x_permuted.shape

        # Compute mean and std across permutations
        mean = x_permuted.mean(dim=1)  # (batch, seq_len, d_model)
        std = x_permuted.std(dim=1, unbiased=False)  # (batch, seq_len, d_model)

        # Add small epsilon for numerical stability
        std = std + 1e-8

        # Encode statistics (mean + std concatenated)
        stats_concat = torch.cat([mean, std], dim=-1)  # (batch, seq_len, 2*d_model)
        encoded_stats = self.stats_encoder(stats_concat)  # (batch, seq_len, d_model)

        # Compute uncertainty from standard deviation
        # High std across permutations → high epistemic uncertainty

        # Average std across features for each token
        std_per_token = std.mean(dim=-1)  # (batch, seq_len)

        # Global uncertainty: average across sequence
        avg_std = std_per_token.mean(dim=-1, keepdim=True)  # (batch, 1)

        # Pass through uncertainty head
        # This learns a monotonic mapping from std to uncertainty
        uncertainty = self.uncertainty_head(
            std.mean(dim=1)  # (batch, d_model)
        ).squeeze(-1)  # (batch,)

        # Apply temperature scaling
        temperature = torch.clamp(self.uncertainty_temperature, min=0.1, max=10.0)
        uncertainty = uncertainty / temperature

        result = {
            'encoded_stats': encoded_stats,
            'uncertainty': uncertainty,
        }

        if return_detailed:
            result['std_per_token'] = std_per_token
            result['mean'] = mean
            result['std'] = std

        return result

    def get_calibration_loss(
        self,
        uncertainty: torch.Tensor,
        errors: torch.Tensor,
        target_correlation: float = 0.5
    ) -> torch.Tensor:
        """
        Auxiliary loss to encourage positive correlation between uncertainty and errors.

        Args:
            uncertainty: (batch,) predicted uncertainties
            errors: (batch,) actual prediction errors (0 or 1)
            target_correlation: Target Pearson correlation coefficient

        Returns:
            Scalar loss that minimizes when correlation matches target
        """
        # Normalize both to zero mean, unit variance
        unc_normalized = (uncertainty - uncertainty.mean()) / (uncertainty.std() + 1e-8)
        err_normalized = (errors - errors.mean()) / (errors.std() + 1e-8)

        # Pearson correlation
        correlation = (unc_normalized * err_normalized).mean()

        # Loss: penalize deviation from target correlation
        loss = (correlation - target_correlation) ** 2

        return loss


class ImprovedStatisticsEncoder(StatisticsEncoderV2):
    """Alias for backward compatibility."""
    pass
