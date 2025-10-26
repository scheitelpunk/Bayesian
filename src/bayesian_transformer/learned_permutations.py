"""
Learned permutation generator using Gumbel-Softmax for differentiable sampling.
Replaces random permutations with learned, meaningful transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LearnedPermutationGenerator(nn.Module):
    """
    Generate learned permutations via Gumbel-Softmax trick.

    Key features:
    1. Differentiable permutation sampling
    2. Learns which input transformations are informative
    3. Regularization to ensure valid, diverse permutations
    4. Temperature annealing for training stability
    """

    def __init__(
        self,
        n_positions: int,
        k_permutations: int,
        temperature: float = 1.0,
        min_temperature: float = 0.3,
        anneal_rate: float = 0.03
    ):
        """
        Args:
            n_positions: Sequence length (e.g., 128)
            k_permutations: Number of permutations (e.g., 5)
            temperature: Initial Gumbel-Softmax temperature
            min_temperature: Minimum temperature during annealing
            anneal_rate: Temperature decay rate per training step
        """
        super().__init__()
        self.n_positions = n_positions
        self.k_permutations = k_permutations
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

        # Learnable logits for each permutation matrix
        # Each permutation is an (n x n) doubly stochastic matrix
        self.perm_logits = nn.Parameter(
            torch.randn(k_permutations, n_positions, n_positions) * 0.1
        )

        # Temperature (can be annealed during training)
        self.register_buffer('temperature', torch.tensor(temperature))
        self.register_buffer('training_step', torch.tensor(0))

    def anneal_temperature(self):
        """Gradually decrease temperature during training."""
        if self.training:
            self.training_step += 1
            new_temp = max(
                self.min_temperature,
                self.temperature * (1 - self.anneal_rate)
            )
            self.temperature.fill_(new_temp)

    def forward(
        self,
        batch_size: int,
        hard: bool = True,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Generate differentiable permutation matrices.

        Args:
            batch_size: Number of samples in batch
            hard: If True, use straight-through estimator (hard forward, soft backward)
            return_logits: If True, also return soft permutations

        Returns:
            perm_matrices: (k, batch, n, n) permutation matrices
        """
        k, n, _ = self.perm_logits.shape

        # Gumbel-Softmax sampling
        # Add Gumbel noise for stochastic sampling
        if self.training:
            # Sample from Gumbel(0, 1)
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(self.perm_logits) + 1e-10
            ) + 1e-10)

            perturbed_logits = (self.perm_logits + gumbel_noise) / self.temperature
        else:
            # Deterministic during inference
            perturbed_logits = self.perm_logits / self.temperature

        # Softmax to get soft permutation matrices (doubly stochastic)
        # Apply softmax on last dimension (each row sums to 1)
        soft_perms = F.softmax(perturbed_logits, dim=-1)  # (k, n, n)

        if hard:
            # Straight-through estimator
            # Forward: hard argmax (one-hot)
            # Backward: gradient from soft_perms
            hard_perms = torch.zeros_like(soft_perms)
            indices = soft_perms.argmax(dim=-1)  # (k, n)
            hard_perms.scatter_(-1, indices.unsqueeze(-1), 1.0)

            # Gradient flows through soft_perms
            perms = hard_perms - soft_perms.detach() + soft_perms
        else:
            perms = soft_perms

        # Expand for batch dimension
        perms = perms.unsqueeze(1).expand(-1, batch_size, -1, -1)  # (k, batch, n, n)

        if return_logits:
            return perms, soft_perms

        return perms

    def get_regularization_loss(self, soft_perms: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Regularization to encourage valid, diverse permutations.

        Components:
        1. Row-stochastic: Each row sums to 1
        2. Column-stochastic: Each column sums to 1 (doubly stochastic)
        3. Diversity: Encourage different permutations to be distinct
        4. Sparsity: Encourage permutations to be close to hard (one-hot rows)

        Args:
            soft_perms: Optional precomputed soft permutations (k, n, n)

        Returns:
            Scalar regularization loss
        """
        if soft_perms is None:
            soft_perms = F.softmax(self.perm_logits, dim=-1)  # (k, n, n)

        k, n, _ = soft_perms.shape

        # 1. Row-stochastic constraint (each row sums to 1)
        row_sums = soft_perms.sum(dim=-1)  # (k, n)
        row_loss = F.mse_loss(row_sums, torch.ones_like(row_sums))

        # 2. Column-stochastic constraint (each column sums to 1)
        col_sums = soft_perms.sum(dim=-2)  # (k, n)
        col_loss = F.mse_loss(col_sums, torch.ones_like(col_sums))

        # 3. Diversity: minimize pairwise similarity
        diversity_loss = 0.0
        if k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    # Cosine similarity between flattened permutation matrices
                    sim = F.cosine_similarity(
                        soft_perms[i].reshape(-1),
                        soft_perms[j].reshape(-1),
                        dim=0
                    )
                    diversity_loss += torch.abs(sim)  # Minimize absolute similarity

            diversity_loss = diversity_loss / (k * (k - 1) / 2)

        # 4. Sparsity: encourage rows to be one-hot (helps with hard permutations)
        # Entropy penalty: low entropy = more peaked distribution
        entropy = -(soft_perms * torch.log(soft_perms + 1e-10)).sum(dim=-1).mean()
        sparsity_loss = entropy  # Minimize entropy

        # Combine losses with weights
        total_loss = (
            1.0 * row_loss +
            1.0 * col_loss +
            0.1 * diversity_loss +
            0.05 * sparsity_loss
        )

        return total_loss

    def get_permutation_quality_metrics(self) -> dict:
        """
        Compute metrics to monitor permutation learning.

        Returns:
            Dictionary with quality metrics
        """
        with torch.no_grad():
            soft_perms = F.softmax(self.perm_logits, dim=-1)  # (k, n, n)

            # How close to hard permutations? (max value in each row)
            max_vals = soft_perms.max(dim=-1)[0]  # (k, n)
            hardness = max_vals.mean().item()  # 1.0 = perfect hard, 1/n = uniform

            # Diversity: average pairwise distance
            k = self.k_permutations
            diversity = 0.0
            if k > 1:
                for i in range(k):
                    for j in range(i + 1, k):
                        dist = (soft_perms[i] - soft_perms[j]).abs().mean()
                        diversity += dist.item()
                diversity = diversity / (k * (k - 1) / 2)

            # Row/column sum deviation
            row_sums = soft_perms.sum(dim=-1)
            col_sums = soft_perms.sum(dim=-2)
            row_deviation = (row_sums - 1.0).abs().mean().item()
            col_deviation = (col_sums - 1.0).abs().mean().item()

            return {
                'hardness': hardness,
                'diversity': diversity,
                'row_deviation': row_deviation,
                'col_deviation': col_deviation,
                'temperature': self.temperature.item()
            }


class HybridPermutationGenerator(nn.Module):
    """
    Hybrid generator: combines learned and random permutations.
    Useful as a fallback if fully learned permutations are unstable.
    """

    def __init__(
        self,
        n_positions: int,
        k_learned: int,
        k_random: int,
        **learned_kwargs
    ):
        super().__init__()
        self.k_learned = k_learned
        self.k_random = k_random
        self.n_positions = n_positions

        # Learned component
        if k_learned > 0:
            self.learned_gen = LearnedPermutationGenerator(
                n_positions, k_learned, **learned_kwargs
            )

    def forward(self, batch_size: int, hard: bool = True) -> torch.Tensor:
        """Generate hybrid permutations."""
        all_perms = []

        # Learned permutations
        if self.k_learned > 0:
            learned_perms = self.learned_gen(batch_size, hard=hard)
            all_perms.append(learned_perms)

        # Random permutations (not differentiable, but stable)
        if self.k_random > 0:
            random_perms = []
            for _ in range(self.k_random):
                perm_indices = torch.randperm(self.n_positions)
                perm_matrix = F.one_hot(
                    perm_indices,
                    num_classes=self.n_positions
                ).float()

                # Expand for batch
                perm_matrix = perm_matrix.unsqueeze(0).expand(batch_size, -1, -1)
                random_perms.append(perm_matrix)

            random_perms = torch.stack(random_perms)  # (k_random, batch, n, n)
            all_perms.append(random_perms)

        # Concatenate
        all_perms = torch.cat(all_perms, dim=0)  # (k_total, batch, n, n)

        return all_perms

    def get_regularization_loss(self) -> torch.Tensor:
        """Only regularize learned component."""
        if self.k_learned > 0:
            return self.learned_gen.get_regularization_loss()
        return torch.tensor(0.0)
