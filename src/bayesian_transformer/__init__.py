"""Bayesian Expectation Transformer Implementation

This package implements the theoretical insights from
"LLMs are Bayesian in Expectation, Not in Realization"
"""

from .bayesian_transformer import (
    BayesianExpectationTransformerLayer,
    MartingaleAwareAttention,
    OptimalCoTLayer,
    SufficientStatsEncoder,
    MDLRegularizedLoss,
    PositionalDebiasing
)

from .checkpointing import (
    CheckpointManager,
    resume_training
)

from .monitoring import (
    BayesianTransformerLogger
)

__version__ = "0.1.0"
__all__ = [
    "BayesianExpectationTransformerLayer",
    "MartingaleAwareAttention",
    "OptimalCoTLayer",
    "SufficientStatsEncoder",
    "MDLRegularizedLoss",
    "PositionalDebiasing",
    "CheckpointManager",
    "resume_training",
    "BayesianTransformerLogger",
]
