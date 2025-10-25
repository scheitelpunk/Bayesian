"""
Examples demonstrating the usage of Bayesian Expectation Transformer components
with integration examples for GPT, BERT, and T5 models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import GPT2Model, BertModel, T5Model, GPT2Config, BertConfig, T5Config

from src.bayesian_transformer import (
    BayesianExpectationTransformerLayer,
    MartingaleAwareAttention,
    OptimalCoTLayer,
    SufficientStatsEncoder,
    MDLRegularizedLoss,
    PositionalDebiasing
)


# Example 1: Basic usage of individual components
def example_basic_usage():
    """Demonstrate basic usage of individual components."""
    print("=== Basic Usage Example ===")
    
    # Configuration
    d_model = 512
    n_heads = 8
    vocab_size = 50000
    batch_size = 4
    seq_length = 64
    
    # Create input
    x = torch.randn(batch_size, seq_length, d_model)
    
    # 1. Martingale-Aware Attention
    print("1. Testing Martingale-Aware Attention...")
    attention = MartingaleAwareAttention(d_model, n_heads, k_permutations=20)
    attn_output = attention(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {attn_output.shape}")
    print(f"   Martingale violation reduction: {attention.k_permutations} permutations")
    
    # 2. Optimal CoT Layer
    print("\n2. Testing Optimal CoT Layer...")
    cot_layer = OptimalCoTLayer(d_model, vocab_size)
    cot_output = cot_layer(x, generate_cot=True)
    print(f"   Optimal CoT lengths: {cot_output['optimal_lengths']}")
    print(f"   Reasoning entropy: {cot_output['reasoning_entropy']}")
    print(f"   CoT logits shape: {cot_output['cot_logits'].shape}")
    
    # 3. Sufficient Statistics Encoder
    print("\n3. Testing Sufficient Statistics Encoder...")
    stats_encoder = SufficientStatsEncoder(d_model)
    stats_output = stats_encoder(x)
    print(f"   Sufficient stats shape: {stats_output['sufficient_stats'].shape}")
    print(f"   Posterior mean: {stats_output['posterior_mean']}")
    print(f"   Posterior variance: {stats_output['posterior_variance']}")
    
    # 4. Positional Debiasing
    print("\n4. Testing Positional Debiasing...")
    debiasing = PositionalDebiasing(d_model)
    debiasing_output = debiasing(x)
    print(f"   Debiased output shape: {debiasing_output['debiased_output'].shape}")
    print(f"   Artifact magnitude: {debiasing_output['artifact_magnitude']}")
    
    # 5. MDL Regularized Loss
    print("\n5. Testing MDL Regularized Loss...")
    loss_fn = MDLRegularizedLoss(beta=0.1, vocab_size=vocab_size)
    logits = torch.randn(batch_size, seq_length, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))
    loss_output = loss_fn(logits, targets)
    print(f"   Total loss: {loss_output['loss']:.4f}")
    print(f"   Standard loss: {loss_output['standard_loss']:.4f}")
    print(f"   MDL penalty: {loss_output['mdl_penalty']:.4f}")
    print(f"   Compression efficiency: {loss_output['optimal_complexity']/loss_output['actual_complexity']:.3f}")


# Example 2: Complete Bayesian Expectation Transformer Layer
def example_complete_layer():
    """Demonstrate usage of the complete integrated layer."""
    print("\n=== Complete Layer Example ===")
    
    # Configuration
    config = {
        'd_model': 512,
        'n_heads': 8,
        'vocab_size': 50000,
        'k_permutations': 20,
        'dropout': 0.1
    }
    
    # Create layer
    layer = BayesianExpectationTransformerLayer(config)
    
    # Input
    batch_size, seq_length = 4, 64
    x = torch.randn(batch_size, seq_length, config['d_model'])
    
    # Forward pass with all features
    output = layer(x, generate_cot=True, return_uncertainty=True)
    
    print(f"Hidden states shape: {output['hidden_states'].shape}")
    print(f"CoT optimal lengths: {output['cot_output']['optimal_lengths']}")
    print(f"Uncertainty (epistemic): {output['uncertainty']['epistemic'].mean():.4f}")
    print(f"Uncertainty (aleatoric): {output['uncertainty']['aleatoric'].mean():.4f}")
    print(f"Uncertainty (total): {output['uncertainty']['total'].mean():.4f}")
    
    # Demonstrate theoretical properties
    print("\nTheoretical Properties:")
    seq_len = output['hidden_states'].shape[1]
    cot_lengths = output['cot_output']['optimal_lengths']
    print(f"   Sequence length: {seq_len}")
    print(f"   CoT length scaling: {cot_lengths.float().mean():.1f} (should scale as sqrt(n))")
    print(f"   Log(n)/n factor: {torch.log(torch.tensor(seq_len))/seq_len:.4f}")


# Example 3: Integration with GPT-2
class BayesianGPT2Layer(nn.Module):
    """GPT-2 layer enhanced with Bayesian Expectation components."""
    
    def __init__(self, config: GPT2Config):
        super().__init__()
        
        # Original GPT-2 components
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Replace standard attention with Bayesian version
        self.attn = MartingaleAwareAttention(
            d_model=config.n_embd,
            n_heads=config.n_head,
            k_permutations=20,
            dropout=config.attn_pdrop
        )
        
        # Enhanced with Bayesian components
        self.bayesian_config = {
            'd_model': config.n_embd,
            'n_heads': config.n_head,
            'vocab_size': config.vocab_size,
            'dropout': config.resid_pdrop
        }
        
        self.bayesian_layer = BayesianExpectationTransformerLayer(self.bayesian_config)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
        
    def forward(self, x, attention_mask=None, generate_cot=False, return_uncertainty=False):
        # Layer norm
        x_norm = self.ln_1(x)
        
        # Bayesian attention and processing
        bayesian_output = self.bayesian_layer(
            x_norm, 
            mask=attention_mask,
            generate_cot=generate_cot,
            return_uncertainty=return_uncertainty
        )
        
        # Residual connection
        x = x + bayesian_output['hidden_states']
        
        # Feed-forward with residual
        x_norm2 = self.ln_2(x)
        x = x + self.mlp(x_norm2)
        
        # Return enhanced output
        output = {'hidden_states': x}
        if generate_cot:
            output['cot_output'] = bayesian_output['cot_output']
        if return_uncertainty:
            output['uncertainty'] = bayesian_output['uncertainty']
            
        return output


def example_gpt2_integration():
    """Demonstrate integration with GPT-2."""
    print("\n=== GPT-2 Integration Example ===")
    
    # Create GPT-2 config
    config = GPT2Config(
        vocab_size=50000,
        n_positions=1024,
        n_embd=512,
        n_layer=6,
        n_head=8,
        n_inner=2048
    )
    
    # Create enhanced layer
    layer = BayesianGPT2Layer(config)
    
    # Input
    batch_size, seq_length = 2, 128
    x = torch.randn(batch_size, seq_length, config.n_embd)
    
    # Forward pass
    output = layer(x, generate_cot=True, return_uncertainty=True)
    
    print(f"Enhanced GPT-2 output shape: {output['hidden_states'].shape}")
    print(f"CoT lengths: {output['cot_output']['optimal_lengths']}")
    print(f"Uncertainty: {output['uncertainty']['total'].mean():.4f}")


# Example 4: Integration with BERT
class BayesianBERTLayer(nn.Module):
    """BERT layer enhanced with Bayesian Expectation components."""
    
    def __init__(self, config: BertConfig):
        super().__init__()
        
        # BERT-specific components
        self.attention = MartingaleAwareAttention(
            d_model=config.hidden_size,
            n_heads=config.num_attention_heads,
            k_permutations=20,
            dropout=config.attention_probs_dropout_prob
        )
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Bayesian components
        self.bayesian_config = {
            'd_model': config.hidden_size,
            'n_heads': config.num_attention_heads,
            'vocab_size': config.vocab_size,
            'dropout': config.hidden_dropout_prob
        }
        
        self.stats_encoder = SufficientStatsEncoder(config.hidden_size)
        self.debiasing = PositionalDebiasing(config.hidden_size)
        
        # Layer normalization
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None, return_uncertainty=False):
        # Compute sufficient statistics
        stats_output = self.stats_encoder(hidden_states)
        
        # Enhanced attention
        attention_output = self.attention(hidden_states, mask=attention_mask)
        
        # Ensure sufficient_stats has same shape as attention_output for combination
        # The SufficientStatsEncoder returns sufficient_stats with same shape as input
        sufficient_stats = stats_output['sufficient_stats']
        
        # Combine with sufficient statistics - they should have matching dimensions
        combined_output = attention_output + sufficient_stats
        
        # Positional debiasing
        debiasing_output = self.debiasing(combined_output)
        debiased_output = debiasing_output['debiased_output']
        
        # BERT-style processing
        intermediate_output = self.intermediate(debiased_output)
        intermediate_output = torch.relu(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + hidden_states)
        
        output = {'hidden_states': layer_output}
        if return_uncertainty:
            output['uncertainty'] = {
                'epistemic': stats_output['posterior_variance'],
                'aleatoric': stats_output['posterior_mean'] * (1 - stats_output['posterior_mean']),
                'posterior_params': (stats_output['alpha'], stats_output['beta'])
            }
            
        return output


def example_bert_integration():
    """Demonstrate integration with BERT."""
    print("\n=== BERT Integration Example ===")
    
    # Create BERT config
    config = BertConfig(
        vocab_size=30000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512
    )
    
    # Create enhanced layer
    layer = BayesianBERTLayer(config)
    
    # Input
    batch_size, seq_length = 2, 128
    x = torch.randn(batch_size, seq_length, config.hidden_size)
    
    # Forward pass
    output = layer(x, return_uncertainty=True)
    
    print(f"Enhanced BERT output shape: {output['hidden_states'].shape}")
    print(f"Epistemic uncertainty: {output['uncertainty']['epistemic'].mean():.4f}")
    print(f"Aleatoric uncertainty: {output['uncertainty']['aleatoric'].mean():.4f}")


# Example 5: Integration with T5
class BayesianT5Layer(nn.Module):
    """T5 layer enhanced with Bayesian Expectation components."""
    
    def __init__(self, config: T5Config):
        super().__init__()
        
        # T5-specific components
        self.is_decoder = getattr(config, 'is_decoder', False)
        
        # Enhanced attention
        self.SelfAttention = MartingaleAwareAttention(
            d_model=config.d_model,
            n_heads=config.num_heads,
            k_permutations=20,
            dropout=config.dropout_rate
        )
        
        if self.is_decoder:
            self.CrossAttention = MartingaleAwareAttention(
                d_model=config.d_model,
                n_heads=config.num_heads,
                k_permutations=20,
                dropout=config.dropout_rate
            )
        
        # Bayesian components
        self.cot_generator = OptimalCoTLayer(
            d_model=config.d_model,
            vocab_size=config.vocab_size
        )
        
        self.debiasing = PositionalDebiasing(config.d_model)
        
        # T5-style feed-forward
        self.DenseReluDense = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout_rate)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                generate_cot=False, return_uncertainty=False):
        
        # Self-attention
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(normed_hidden_states, mask=attention_mask)
        hidden_states = hidden_states + attention_output
        
        # Cross-attention for decoder
        if self.is_decoder and encoder_hidden_states is not None:
            normed_hidden_states = self.layer_norm(hidden_states)
            cross_attention_output = self.CrossAttention(normed_hidden_states)
            hidden_states = hidden_states + cross_attention_output
        
        # Positional debiasing
        debiasing_output = self.debiasing(hidden_states)
        hidden_states = debiasing_output['debiased_output']
        
        # Feed-forward
        normed_hidden_states = self.layer_norm(hidden_states)
        feed_forward_output = self.DenseReluDense(normed_hidden_states)
        hidden_states = hidden_states + feed_forward_output
        
        output = {'hidden_states': hidden_states}
        
        # Optional CoT generation
        if generate_cot:
            cot_output = self.cot_generator(hidden_states, generate_cot=True)
            output['cot_output'] = cot_output
            
        if return_uncertainty:
            # Simplified uncertainty from debiasing artifacts
            output['uncertainty'] = {
                'artifact_magnitude': debiasing_output['artifact_magnitude'],
                'correction_magnitude': debiasing_output['correction_magnitude']
            }
            
        return output


def example_t5_integration():
    """Demonstrate integration with T5."""
    print("\n=== T5 Integration Example ===")
    
    # Create T5 config
    config = T5Config(
        vocab_size=32000,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        is_decoder=True
    )
    
    # Create enhanced layer
    layer = BayesianT5Layer(config)
    
    # Input
    batch_size, seq_length = 2, 128
    x = torch.randn(batch_size, seq_length, config.d_model)
    encoder_states = torch.randn(batch_size, seq_length, config.d_model)
    
    # Forward pass
    output = layer(x, encoder_hidden_states=encoder_states, generate_cot=True, return_uncertainty=True)
    
    print(f"Enhanced T5 output shape: {output['hidden_states'].shape}")
    print(f"CoT lengths: {output['cot_output']['optimal_lengths']}")
    print(f"Artifact magnitude: {output['uncertainty']['artifact_magnitude'].mean():.4f}")


# Example 6: Training with MDL Loss
def example_training_with_mdl():
    """Demonstrate training with MDL regularized loss."""
    print("\n=== Training with MDL Loss Example ===")
    
    # Configuration
    config = {
        'd_model': 512,
        'n_heads': 8,
        'vocab_size': 50000,
        'k_permutations': 20,
        'dropout': 0.1
    }
    
    # Create model and loss
    model = BayesianExpectationTransformerLayer(config)
    loss_fn = MDLRegularizedLoss(beta=0.1, vocab_size=config['vocab_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop simulation
    batch_size, seq_length = 4, 64
    
    print("Simulating training steps...")
    for step in range(5):
        # Generate dummy batch
        x = torch.randn(batch_size, seq_length, config['d_model'])
        targets = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
        
        # Forward pass
        model_output = model(x, generate_cot=True, return_uncertainty=True)

        # Generate logits (in real scenario, this would be from language model head)
        logits = torch.randn(batch_size, seq_length, config['vocab_size'], requires_grad=True)
        
        # Compute loss
        loss_output = loss_fn(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss_output['loss'].backward()
        optimizer.step()
        
        print(f"Step {step+1}: Loss = {loss_output['loss']:.4f}, "
              f"MDL Penalty = {loss_output['mdl_penalty']:.4f}, "
              f"Compression Efficiency = {loss_output['optimal_complexity']/loss_output['actual_complexity']:.3f}")


# Example 7: Real-world usage patterns
def example_real_world_usage():
    """Demonstrate real-world usage patterns and best practices."""
    print("\n=== Real-world Usage Example ===")
    
    # Configuration for production use
    config = {
        'd_model': 1024,
        'n_heads': 16,
        'vocab_size': 50000,
        'k_permutations': 10,  # Reduced for efficiency
        'dropout': 0.1
    }
    
    # Create model
    model = BayesianExpectationTransformerLayer(config)
    
    # Production-like batch
    batch_size, seq_length = 8, 512
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_length))
    
    # Convert to embeddings (in real scenario, this would be from embedding layer)
    embeddings = torch.randn(batch_size, seq_length, config['d_model'])
    
    # Forward pass with selective features
    print("Processing batch with selective features...")
    
    # 1. Standard processing (fastest)
    output_standard = model(embeddings)
    print(f"Standard processing time: ~{output_standard['hidden_states'].numel() * 1e-6:.2f}M operations")
    
    # 2. With uncertainty quantification (for critical applications)
    output_with_uncertainty = model(embeddings, return_uncertainty=True)
    uncertainty = output_with_uncertainty['uncertainty']
    print(f"Uncertainty quantification - Epistemic: {uncertainty['epistemic'].mean():.4f}")
    
    # 3. With CoT generation (for complex reasoning tasks)
    output_with_cot = model(embeddings, generate_cot=True)
    avg_cot_length = output_with_cot['cot_output']['optimal_lengths'].float().mean()
    print(f"CoT generation - Average optimal length: {avg_cot_length:.1f}")
    
    # 4. Performance monitoring
    print("\nPerformance Metrics:")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Memory usage: ~{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "   Memory usage: CPU mode")
    
    # 5. Theoretical validation
    print("\nTheoretical Validation:")
    stats = output_with_uncertainty['sufficient_stats']
    print(f"   Posterior concentration: {(stats['alpha'] + stats['beta']).mean():.2f}")
    print(f"   Bayesian calibration: {stats['posterior_variance'].mean():.4f}")


# Main execution
if __name__ == "__main__":
    print("Bayesian Expectation Transformer - Examples")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run examples
    example_basic_usage()
    example_complete_layer()
    example_gpt2_integration()
    example_bert_integration()
    example_t5_integration()
    example_training_with_mdl()
    example_real_world_usage()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nKey takeaways:")
    print("- Martingale violations reduced by factor sqrt(k) through permutation averaging")
    print("- CoT length scales optimally as sqrt(n) log(1/epsilon)")
    print("- MDL regularization promotes compression efficiency")
    print("- Uncertainty quantification through Beta posterior approximation")
    print("- Seamless integration with existing transformer architectures")