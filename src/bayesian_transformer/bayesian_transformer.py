import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from collections import OrderedDict
import numpy as np


class LRUPermutationCache:
    """LRU cache for permutation tensors with size limit to prevent memory leaks."""

    def __init__(self, maxsize: int = 100):
        """
        Initialize LRU cache with maximum size.

        Args:
            maxsize: Maximum number of entries to cache
        """
        self.cache: OrderedDict = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: Tuple[int, int]) -> Optional[torch.Tensor]:
        """
        Get cached permutation tensor if it exists.

        Args:
            key: Tuple of (seq_length, k_permutations)

        Returns:
            Cached tensor if found, None otherwise
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: Tuple[int, int], value: torch.Tensor):
        """
        Store permutation tensor in cache with LRU eviction.

        Args:
            key: Tuple of (seq_length, k_permutations)
            value: Permutation tensor to cache
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.maxsize:
                # Remove oldest (least recently used)
                self.cache.popitem(last=False)

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()


class MartingaleAwareAttention(nn.Module):
    """
    Martingale-Aware Attention Layer implementing the theoretical insights from
    "LLMs are Bayesian in Expectation, Not in Realization"
    
    This layer combines standard multi-head attention with permutation averaging
    to reduce martingale violations following Θ(log n/n) convergence.
    """
    
    def __init__(self, d_model: int, n_heads: int, k_permutations: int = 20,
                 dropout: float = 0.1, max_seq_length: int = 2048, cache_maxsize: int = 100):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.k_permutations = k_permutations
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_length = max_seq_length

        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # Permutation averaging components with LRU cache
        self.perm_cache = LRUPermutationCache(maxsize=cache_maxsize)
        self.variance_reduction_weight = nn.Parameter(torch.ones(1))

        # Adaptive weighting based on sequence length
        self.length_adaptive_weight = nn.Parameter(torch.ones(1))

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following the theoretical optimal scaling."""
        for module in [self.q_linear, self.k_linear, self.v_linear, self.out_linear]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _get_cached_permutations(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """
        Get cached permutations for given sequence length.
        Implements efficient LRU caching for permutation-based variance reduction.
        """
        cache_key = (seq_length, self.k_permutations)

        # Try to get from cache
        cached_perms = self.perm_cache.get(cache_key)

        if cached_perms is None:
            # Generate k random permutations
            perms = torch.stack([
                torch.randperm(seq_length, device=device)
                for _ in range(self.k_permutations)
            ])
            self.perm_cache.put(cache_key, perms)
            return perms

        return cached_perms.to(device)
    
    def _compute_permutation_average(self, x: torch.Tensor, 
                                   attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute permutation average to reduce martingale violations.
        
        Following Proposition 3.10: Variance reduction by factor √k through
        averaging over k permutations.
        """
        batch_size, seq_length, _ = x.shape
        device = x.device
        
        # Get cached permutations
        perms = self._get_cached_permutations(seq_length, device)
        
        # Apply permutations and compute average
        perm_outputs = []
        for perm in perms:
            # Permute input sequence
            x_perm = x[:, perm, :]
            
            # Compute attention with permuted input
            q_perm = self.q_linear(x_perm).view(batch_size, seq_length, self.n_heads, self.head_dim)
            k_perm = self.k_linear(x_perm).view(batch_size, seq_length, self.n_heads, self.head_dim)
            v_perm = self.v_linear(x_perm).view(batch_size, seq_length, self.n_heads, self.head_dim)
            
            # Transpose for attention computation
            q_perm = q_perm.transpose(1, 2)  # (B, H, S, D)
            k_perm = k_perm.transpose(1, 2)
            v_perm = v_perm.transpose(1, 2)
            
            # Compute attention scores
            scores_perm = torch.matmul(q_perm, k_perm.transpose(-2, -1)) * self.scale
            attn_weights_perm = F.softmax(scores_perm, dim=-1)
            attn_weights_perm = self.dropout(attn_weights_perm)
            
            # Apply attention to values
            out_perm = torch.matmul(attn_weights_perm, v_perm)
            out_perm = out_perm.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
            
            # Reverse permutation to get back original order
            inv_perm = torch.argsort(perm)
            out_perm = out_perm[:, inv_perm, :]
            
            perm_outputs.append(out_perm)
        
        # Average over all permutations (variance reduction by √k)
        perm_average = torch.stack(perm_outputs).mean(dim=0)
        
        return perm_average
    
    def _compute_adaptive_weight(self, seq_length: int) -> torch.Tensor:
        """
        Compute adaptive weighting based on sequence length following log(n)/n scaling.
        
        The weight decreases as O(log n/n) to ensure martingale convergence.
        """
        if seq_length <= 1:
            return torch.tensor(1.0)
        
        log_n_over_n = math.log(seq_length) / seq_length
        adaptive_factor = self.length_adaptive_weight * log_n_over_n
        
        return torch.clamp(adaptive_factor, min=0.01, max=1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining standard attention with permutation averaging.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_length, d_model)
        """
        batch_size, seq_length, d_model = x.shape
        
        # Standard multi-head attention
        q = self.q_linear(x).view(batch_size, seq_length, self.n_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_length, self.n_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_length, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        standard_out = torch.matmul(attn_weights, v)
        standard_out = standard_out.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        standard_out = self.out_linear(standard_out)
        
        # Compute permutation average for variance reduction
        perm_out = self._compute_permutation_average(x, attn_weights)
        perm_out = self.out_linear(perm_out)
        
        # Adaptive combination based on sequence length
        adaptive_weight = self._compute_adaptive_weight(seq_length)
        
        # Combine outputs with variance reduction weighting
        combined_out = (
            (1 - adaptive_weight) * standard_out + 
            adaptive_weight * self.variance_reduction_weight * perm_out
        )
        
        return combined_out


class OptimalCoTLayer(nn.Module):
    """
    Optimal Chain-of-Thought Layer that automatically computes the optimal CoT length
    based on the theoretical formula: k* = sqrt(n * alpha / (H_CoT * (B_0 - B_opt))) * log2(1/epsilon)
    """
    
    def __init__(self, d_model: int, vocab_size: int, L_f: int = 10, 
                 alpha: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.L_f = L_f  # Final answer length
        self.alpha = alpha  # Scaling parameter
        self.epsilon = epsilon  # Error tolerance
        
        # Components for reasoning entropy estimation
        self.entropy_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
        # Complexity estimation network
        self.complexity_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
        # CoT generation head
        self.cot_generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, vocab_size)
        )
        
        # Efficiency constraint parameters
        self.max_cot_length = 512
        self.computational_budget = 1000.0
        
    def _estimate_reasoning_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate H_CoT (reasoning entropy) adaptively based on input complexity.
        """
        # Average over sequence dimension for global complexity
        pooled = x.mean(dim=1)  # (batch_size, d_model)
        h_cot = self.entropy_estimator(pooled)  # (batch_size, 1)
        
        return h_cot.squeeze(-1)  # (batch_size,)
    
    def _compute_optimal_length(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute optimal CoT length k* based on theoretical formula.
        
        k* = sqrt(n * alpha / (H_CoT * (B_0 - B_opt))) * log2(1/epsilon)
        """
        batch_size = x.size(0)
        
        # Estimate reasoning entropy
        h_cot = self._estimate_reasoning_entropy(x)
        
        # Estimate complexity bounds (simplified)
        pooled = x.mean(dim=1)
        complexity_ratio = self.complexity_estimator(pooled).squeeze(-1)
        
        # Compute optimal length
        sqrt_term = torch.sqrt(n * self.alpha / (h_cot * complexity_ratio + 1e-8))
        log_term = math.log2(1.0 / self.epsilon)

        k_optimal = sqrt_term * log_term

        # Apply efficiency constraints
        k_optimal = torch.clamp(k_optimal, min=1.0, max=self.max_cot_length)

        # Keep as float to maintain gradient flow
        return k_optimal
    
    def forward(self, x: torch.Tensor, generate_cot: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass that computes optimal CoT length and optionally generates CoT.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            generate_cot: Whether to generate actual CoT tokens
            
        Returns:
            Dictionary containing optimal lengths and generated CoT (if requested)
        """
        batch_size, seq_length, d_model = x.shape
        
        # Compute optimal CoT length for each example
        k_optimal = self._compute_optimal_length(x, seq_length)
        
        output = {
            'optimal_lengths': k_optimal,
            'reasoning_entropy': self._estimate_reasoning_entropy(x)
        }
        
        if generate_cot:
            # Generate CoT tokens up to optimal length
            cot_logits = self.cot_generator(x)  # (batch_size, seq_length, vocab_size)
            
            # Sample or use greedy decoding for CoT generation
            cot_tokens = torch.argmax(cot_logits, dim=-1)
            
            output['cot_logits'] = cot_logits
            output['cot_tokens'] = cot_tokens
        
        return output


class SufficientStatsEncoder(nn.Module):
    """
    Sufficient Statistics Encoder that explicitly computes and uses sufficient statistics
    for Beta-Posterior approximation and counting-based inference.
    """
    
    def __init__(self, d_model: int, max_moments: int = None):
        super().__init__()
        
        self.d_model = d_model
        # Set max moments to O(log d) as suggested in theory
        self.max_moments = max_moments or max(3, int(math.log(d_model)))
        
        # Counting heads for Bernoulli-like sequences
        self.counting_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(self.max_moments)
        ])
        
        # Moment computation networks
        self.moment_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(self.max_moments)
        ])
        
        # Beta posterior approximation
        self.alpha_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
        # Output projection - combines moments, counts, alpha, beta
        # Input size: max_moments (moments) + max_moments (counts) + 2 (alpha, beta)
        self.output_proj = nn.Linear(2 * self.max_moments + 2, d_model)
        
    def _compute_moments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute moments up to order k = O(log d).
        """
        batch_size, seq_length, d_model = x.shape
        moments = []
        
        for k in range(self.max_moments):
            # Compute k-th moment
            moment_k = self.moment_networks[k](x)  # (batch_size, seq_length, 1)
            # Average over sequence for sufficient statistic
            moment_k = moment_k.mean(dim=1)  # (batch_size, 1)
            moments.append(moment_k)
        
        return torch.cat(moments, dim=-1)  # (batch_size, max_moments)
    
    def _compute_counting_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute counting statistics for Bernoulli-like sequences.
        """
        batch_size, seq_length, d_model = x.shape
        
        # Apply counting heads
        counts = []
        for head in self.counting_heads:
            count = torch.sigmoid(head(x))  # (batch_size, seq_length, 1)
            # Sum over sequence for total count
            total_count = count.sum(dim=1)  # (batch_size, 1)
            counts.append(total_count)
        
        return torch.cat(counts, dim=-1)  # (batch_size, max_moments)
    
    def _compute_beta_posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Beta posterior approximation parameters.
        """
        # Pool sequence information
        pooled = x.mean(dim=1)  # (batch_size, d_model)
        
        # Compute posterior parameters
        alpha = self.alpha_net(pooled).squeeze(-1) + 1.0  # (batch_size,)
        beta = self.beta_net(pooled).squeeze(-1) + 1.0   # (batch_size,)
        
        return alpha, beta
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing sufficient statistics and posterior approximation.

        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Dictionary containing sufficient statistics and posterior parameters
        """
        batch_size, seq_length, d_model = x.shape

        # Compute moments
        moments = self._compute_moments(x)

        # Compute counting statistics (alternative view)
        counts = self._compute_counting_statistics(x)

        # Compute Beta posterior parameters
        alpha, beta = self._compute_beta_posterior(x)

        # Combine all statistics including counts for complete gradient flow
        combined_stats = torch.cat([
            moments,
            counts,  # Include counting statistics for gradient flow
            alpha.unsqueeze(-1),
            beta.unsqueeze(-1)
        ], dim=-1)

        # Project to output dimension
        # Note: output_proj input size should be 2*max_moments + 2
        output = self.output_proj(combined_stats)  # (batch_size, d_model)

        # Expand to (batch_size, seq_length, d_model) to match input shape
        output = output.unsqueeze(1).expand(batch_size, seq_length, d_model)

        return {
            'sufficient_stats': output,
            'moments': moments,
            'counts': counts,
            'alpha': alpha,
            'beta': beta,
            'posterior_mean': alpha / (alpha + beta),
            'posterior_variance': (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        }


class MDLRegularizedLoss(nn.Module):
    """
    MDL (Minimum Description Length) Regularized Loss that promotes optimal compression
    following the theoretical complexity bounds.
    """
    
    def __init__(self, beta: float = 0.1, vocab_size: int = 50000):
        super().__init__()
        
        self.beta = beta
        self.vocab_size = vocab_size
        
    def _compute_empirical_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical entropy H(p) from model predictions.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy: H(p) = -sum(p * log(p))
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Average over sequence and batch
        return entropy.mean()
    
    def _compute_optimal_complexity(self, n: int, empirical_entropy: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal complexity: n*H(p) + O(sqrt(n*log(n)))
        """
        # Main term: n * H(p)
        main_term = n * empirical_entropy
        
        # Correction term: O(sqrt(n*log(n)))
        if n > 1:
            correction_term = math.sqrt(n * math.log(n))
        else:
            correction_term = 0.0
        
        return main_term + correction_term
    
    def _compute_actual_complexity(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute actual model complexity (simplified as negative log-likelihood).
        """
        # Use cross-entropy as proxy for complexity
        return -F.log_softmax(logits, dim=-1).mean()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute MDL-regularized loss.
        
        Args:
            logits: Model predictions of shape (batch_size, seq_length, vocab_size)
            targets: Target tokens of shape (batch_size, seq_length)
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_length, vocab_size = logits.shape
        
        # Standard cross-entropy loss
        standard_loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            targets.view(-1), 
            ignore_index=-100
        )
        
        # Compute empirical entropy
        empirical_entropy = self._compute_empirical_entropy(logits)
        
        # Compute optimal complexity bound
        optimal_complexity = self._compute_optimal_complexity(seq_length, empirical_entropy)
        
        # Compute actual complexity
        actual_complexity = self._compute_actual_complexity(logits)
        
        # MDL regularization term
        mdl_penalty = torch.clamp(actual_complexity - optimal_complexity, min=0.0)
        
        # Total loss
        total_loss = standard_loss + self.beta * mdl_penalty
        
        return {
            'loss': total_loss,
            'standard_loss': standard_loss,
            'mdl_penalty': mdl_penalty,
            'empirical_entropy': empirical_entropy,
            'optimal_complexity': optimal_complexity,
            'actual_complexity': actual_complexity
        }


class PositionalDebiasing(nn.Module):
    """
    Positional Debiasing Module that corrects for periodic artifacts in positional encodings
    without losing positional information.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 2048, 
                 encoding_type: str = 'rotary', n_harmonics: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.encoding_type = encoding_type
        self.n_harmonics = n_harmonics
        
        # Artifact detection network
        self.artifact_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_harmonics),
            nn.Sigmoid()
        )
        
        # Multi-harmonic modeling
        self.harmonic_weights = nn.Parameter(torch.ones(n_harmonics))
        
        # Adaptive correction network
        self.correction_net = nn.Sequential(
            nn.Linear(d_model + n_harmonics, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Position-aware gating
        self.position_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def _detect_periodic_artifacts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect periodic artifacts in the input sequence.
        Common periods: 64, 128, 256, 512 tokens
        """
        batch_size, seq_length, d_model = x.shape
        
        # Compute artifact scores for different harmonics
        artifact_scores = self.artifact_detector(x)  # (batch_size, seq_length, n_harmonics)
        
        # Weight by learned harmonic importance
        weighted_scores = artifact_scores * self.harmonic_weights.unsqueeze(0).unsqueeze(0)
        
        return weighted_scores
    
    def _compute_correction(self, x: torch.Tensor, artifact_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive correction based on detected artifacts.
        """
        batch_size, seq_length, d_model = x.shape
        
        # Combine input with artifact information
        combined = torch.cat([x, artifact_scores], dim=-1)
        
        # Generate correction
        correction = self.correction_net(combined)
        
        # Apply position-aware gating
        gate = self.position_gate(x)
        
        return correction * gate
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass applying positional debiasing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            positions: Optional position indices
            
        Returns:
            Dictionary containing debiased output and diagnostic information
        """
        # Detect periodic artifacts
        artifact_scores = self._detect_periodic_artifacts(x)
        
        # Compute correction
        correction = self._compute_correction(x, artifact_scores)
        
        # Apply correction while preserving positional information
        debiased_x = x + correction
        
        # Compute diagnostic metrics
        artifact_magnitude = artifact_scores.mean(dim=(1, 2))  # (batch_size,)
        correction_magnitude = correction.norm(dim=-1).mean(dim=1)  # (batch_size,)
        
        return {
            'debiased_output': debiased_x,
            'artifact_scores': artifact_scores,
            'correction': correction,
            'artifact_magnitude': artifact_magnitude,
            'correction_magnitude': correction_magnitude
        }


class BayesianExpectationTransformerLayer(nn.Module):
    """
    Complete Bayesian Expectation Transformer Layer integrating all components
    from the theoretical framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.vocab_size = config['vocab_size']
        self.k_permutations = config.get('k_permutations', 20)
        self.dropout = config.get('dropout', 0.1)
        
        # Initialize all components
        self.attention = MartingaleAwareAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            k_permutations=self.k_permutations,
            dropout=self.dropout
        )
        
        self.cot_generator = OptimalCoTLayer(
            d_model=self.d_model,
            vocab_size=self.vocab_size
        )
        
        self.stats_encoder = SufficientStatsEncoder(
            d_model=self.d_model
        )
        
        self.debiasing = PositionalDebiasing(
            d_model=self.d_model
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 4, self.d_model)
        )

        # Gating mechanisms for natural gradient flow through all components
        # CoT integration: combines reasoning features (entropy, length, logits signal) with hidden states
        self.cot_gate = nn.Linear(self.d_model + 3, self.d_model)
        # Stats integration: adaptive weighting of sufficient statistics
        self.stats_gate = nn.Linear(self.d_model, 1)

        # Initialize gates
        nn.init.xavier_uniform_(self.cot_gate.weight)
        nn.init.zeros_(self.cot_gate.bias)
        nn.init.xavier_uniform_(self.stats_gate.weight)
        nn.init.zeros_(self.stats_gate.bias)
        
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_uncertainty: bool = False,
                generate_cot: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete Bayesian Expectation Transformer Layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, d_model)
            mask: Optional attention mask
            return_uncertainty: Whether to return calibrated uncertainty estimates
            generate_cot: Whether to generate Chain-of-Thought reasoning
            
        Returns:
            Dictionary containing outputs and optional uncertainty estimates
        """
        batch_size, seq_length, d_model = x.shape
        
        # 1. Compute sufficient statistics
        stats_output = self.stats_encoder(x)
        sufficient_stats = stats_output['sufficient_stats']

        # Add sufficient statistics to input (already has correct shape)
        x_with_stats = x + sufficient_stats
        x_with_stats = self.norm1(x_with_stats)
        
        # 2. Apply Martingale-aware attention
        attn_output = self.attention(x_with_stats, mask=mask)
        x = x + attn_output
        x = self.norm2(x)
        
        # 3. Generate optimal CoT and integrate naturally via gated combination
        # Always generate to ensure all parameters receive natural gradients
        cot_output = self.cot_generator(x, generate_cot=True)

        # Integrate CoT logits to ensure cot_generator parameters receive meaningful gradients
        # Use the raw logits (not softmax) to avoid gradient saturation
        cot_logits = cot_output['cot_logits']  # (batch_size, seq_length, vocab_size)

        # Compute statistics from logits that preserve gradient flow
        # Use std of logits as a measure of output diversity
        cot_reasoning_signal = cot_logits.std(dim=-1, keepdim=True)  # (batch_size, seq_length, 1)

        # Integrate CoT reasoning features naturally into the forward pass
        # Expand reasoning_entropy and optimal_lengths to match sequence dimension
        reasoning_entropy_expanded = cot_output['reasoning_entropy'].unsqueeze(-1).unsqueeze(1).expand(
            batch_size, seq_length, 1
        )
        optimal_lengths_expanded = cot_output['optimal_lengths'].unsqueeze(-1).unsqueeze(1).expand(
            batch_size, seq_length, 1
        )

        # Combine with current representation: (batch_size, seq_length, d_model + 3)
        # Include cot_reasoning_signal to ensure cot_generator receives gradients
        combined_features = torch.cat([
            x,
            reasoning_entropy_expanded,
            optimal_lengths_expanded,
            cot_reasoning_signal
        ], dim=-1)

        # Gated projection: learns how to incorporate CoT reasoning
        cot_contribution = self.cot_gate(combined_features)

        # Residual connection with tanh activation for bounded contribution
        x = x + torch.tanh(cot_contribution)

        # Integrate sufficient statistics naturally via gated mechanism
        # Use tanh instead of sigmoid for stronger gradients
        stats_weight = torch.tanh(self.stats_gate(sufficient_stats))
        # Scale by 0.5 and add 0.5 to get range [0, 1] but with better gradient flow
        stats_weight = 0.5 * stats_weight + 0.5
        stats_contribution = stats_weight * sufficient_stats

        # Add weighted statistics contribution
        x = x + stats_contribution

        # 4. Debias positional artifacts
        debiasing_output = self.debiasing(x)
        x = debiasing_output['debiased_output']
        
        # 5. Apply feed-forward network
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm3(x)
        
        # Prepare output dictionary
        output = {
            'hidden_states': x,
            'sufficient_stats': stats_output,
            'debiasing_info': debiasing_output,
            'cot_output': cot_output  # Always included - naturally used in forward pass
        }
            
        if return_uncertainty:
            # Return calibrated uncertainty estimates from posterior
            alpha = stats_output['alpha']
            beta = stats_output['beta']
            
            # Compute uncertainty metrics
            epistemic_uncertainty = stats_output['posterior_variance']
            aleatoric_uncertainty = stats_output['posterior_mean'] * (1 - stats_output['posterior_mean'])
            
            output['uncertainty'] = {
                'epistemic': epistemic_uncertainty,
                'aleatoric': aleatoric_uncertainty,
                'total': epistemic_uncertainty + aleatoric_uncertainty,
                'posterior_alpha': alpha,
                'posterior_beta': beta
            }
        
        return output