from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional
import torch
from datetime import datetime
from pathlib import Path


class BayesianTransformerLogger:
    """TensorBoard logger for Bayesian Transformer training.

    Logs:
    - Training/validation metrics (loss, accuracy)
    - Gradient flow (detect vanishing/exploding)
    - Attention statistics (entropy, weights)
    - Uncertainty metrics (epistemic, aleatoric)
    - Learning rate schedules
    - Model parameters histograms
    """

    def __init__(
        self,
        log_dir: str = 'runs',
        experiment_name: Optional[str] = None,
        log_gradients_every: int = 100,
        log_histograms_every: int = 500
    ):
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(str(self.log_dir))

        self.log_gradients_every = log_gradients_every
        self.log_histograms_every = log_histograms_every
        self.global_step = 0

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = 'train'
    ):
        """Log scalar metrics (loss, accuracy, etc.)."""
        step = step if step is not None else self.global_step

        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)

    def log_gradients(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ):
        """Log gradient statistics to detect vanishing/exploding gradients."""
        if step is None:
            step = self.global_step

        if step % self.log_gradients_every != 0:
            return

        gradient_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[f'gradients/{name}'] = grad_norm

                # Log layer-wise gradient norms
                layer_name = name.split('.')[0]
                if f'gradients/layer_{layer_name}' not in gradient_norms:
                    gradient_norms[f'gradients/layer_{layer_name}'] = []
                gradient_norms[f'gradients/layer_{layer_name}'].append(grad_norm)

        # Log individual gradients
        for name, norm in gradient_norms.items():
            if isinstance(norm, list):
                avg_norm = sum(norm) / len(norm)
                self.writer.add_scalar(name, avg_norm, step)
            else:
                self.writer.add_scalar(name, norm, step)

        # Log overall gradient norm
        total_norm = sum(n for n in gradient_norms.values() if isinstance(n, float))
        self.writer.add_scalar('gradients/total_norm', total_norm, step)

    def log_attention_stats(
        self,
        attention_weights: torch.Tensor,
        attention_entropy: torch.Tensor,
        step: Optional[int] = None,
        prefix: str = 'attention'
    ):
        """Log attention statistics to detect attention collapse."""
        step = step if step is not None else self.global_step

        # Average entropy across batch
        avg_entropy = attention_entropy.mean().item()
        self.writer.add_scalar(f'{prefix}/entropy', avg_entropy, step)

        # Attention weight statistics
        self.writer.add_scalar(
            f'{prefix}/weights_mean',
            attention_weights.mean().item(),
            step
        )
        self.writer.add_scalar(
            f'{prefix}/weights_std',
            attention_weights.std().item(),
            step
        )

    def log_uncertainty_metrics(
        self,
        epistemic_uncertainty: torch.Tensor,
        aleatoric_uncertainty: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ):
        """Log uncertainty quantification metrics."""
        step = step if step is not None else self.global_step

        self.writer.add_scalar(
            'uncertainty/epistemic',
            epistemic_uncertainty.mean().item(),
            step
        )

        if aleatoric_uncertainty is not None:
            self.writer.add_scalar(
                'uncertainty/aleatoric',
                aleatoric_uncertainty.mean().item(),
                step
            )

        # Log histograms
        self.writer.add_histogram(
            'uncertainty/epistemic_dist',
            epistemic_uncertainty,
            step
        )

    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        step: Optional[int] = None
    ):
        """Log current learning rate."""
        step = step if step is not None else self.global_step

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f'learning_rate/group_{i}', lr, step)

    def log_model_histograms(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ):
        """Log parameter and gradient histograms."""
        if step is None:
            step = self.global_step

        if step % self.log_histograms_every != 0:
            return

        for name, param in model.named_parameters():
            # Parameter values
            self.writer.add_histogram(f'params/{name}', param, step)

            # Gradients
            if param.grad is not None:
                self.writer.add_histogram(
                    f'gradients_hist/{name}',
                    param.grad,
                    step
                )

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text (useful for predictions, errors, etc.)."""
        step = step if step is not None else self.global_step
        self.writer.add_text(tag, text, step)

    def increment_step(self):
        """Increment global step counter."""
        self.global_step += 1

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
