"""
DEMO 2: Gradient Flow Visualization Through 20-Layer Network

This demo shows how gradients flow (or vanish) through deep networks
with different activation functions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────
# 1. CREATE DEEP NETWORKS
# ──────────────────────────────────────
def create_deep_network(activation_fn, num_layers=20, hidden_size=64):
    """Create a deep network with specified activation"""
    layers = []
    
    # Input layer
    layers.append(nn.Linear(10, hidden_size))
    layers.append(activation_fn())
    
    # Hidden layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation_fn())
    
    # Output layer
    layers.append(nn.Linear(hidden_size, 1))
    
    return nn.Sequential(*layers)

# ──────────────────────────────────────
# 2. INITIALIZE NETWORKS PROPERLY
# ──────────────────────────────────────
def initialize_network(model, activation_name):
    """Initialize weights based on activation function"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if activation_name in ['ReLU', 'LeakyReLU', 'GELU', 'SiLU']:
                # He initialization for ReLU-like activations
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif activation_name in ['Tanh', 'Sigmoid']:
                # Xavier initialization for sigmoid/tanh
                nn.init.xavier_normal_(module.weight)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)

# ──────────────────────────────────────
# 3. MEASURE GRADIENT FLOW
# ──────────────────────────────────────
def measure_gradient_flow(model, activation_name):
    """
    Measure gradient magnitudes at each layer after backprop.
    Returns layer indices and gradient magnitudes.
    """
    # Create dummy data
    batch_size = 64
    x = torch.randn(batch_size, 10, requires_grad=True)
    target = torch.randn(batch_size, 1)
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradient magnitudes
    gradients = []
    layer_names = []
    layer_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            grad_magnitude = param.grad.abs().mean().item()
            gradients.append(grad_magnitude)
            layer_names.append(f"L{layer_count}")
            layer_count += 1
    
    return layer_names, gradients

# ──────────────────────────────────────
# 4. MEASURE ACTIVATION STATISTICS
# ──────────────────────────────────────
def measure_activation_stats(model, activation_name):
    """Measure activation statistics (mean, std) through the network"""
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().clone())
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, 
                               nn.Tanh, nn.Sigmoid, nn.Mish, nn.ELU)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # Forward pass
    x = torch.randn(64, 10)
    model.eval()
    with torch.no_grad():
        model(x)
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate statistics
    means = [act.mean().item() for act in activations]
    stds = [act.std().item() for act in activations]
    
    return means, stds

# ──────────────────────────────────────
# 5. RUN EXPERIMENTS
# ──────────────────────────────────────
def run_gradient_flow_experiment():
    """Run gradient flow experiment for multiple activations"""
    
    print("="*70)
    print("DEMO 2: Gradient Flow Visualization Through Deep Networks")
    print("="*70)
    
    activations = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
        'ELU': nn.ELU,
    }
    
    num_layers = 20
    results = {}
    
    print(f"\nCreating {num_layers}-layer networks...")
    print("Measuring gradient flow for each activation function...\n")
    
    for name, activation_fn in activations.items():
        print(f"Testing {name}...")
        
        # Create and initialize model
        model = create_deep_network(activation_fn, num_layers=num_layers)
        initialize_network(model, name)
        
        # Measure gradients
        layer_names, gradients = measure_gradient_flow(model, name)
        
        # Measure activations
        means, stds = measure_activation_stats(model, name)
        
        results[name] = {
            'gradients': gradients,
            'layer_names': layer_names,
            'activation_means': means,
            'activation_stds': stds
        }
        
        # Print summary
        first_layer_grad = gradients[-1]  # Input layer
        last_layer_grad = gradients[0]     # Output layer
        ratio = first_layer_grad / last_layer_grad if last_layer_grad > 0 else 0
        
        print(f"  Output layer gradient: {last_layer_grad:.6f}")
        print(f"  Input layer gradient:  {first_layer_grad:.6f}")
        print(f"  Ratio (input/output):  {ratio:.2e}")
        
        if ratio < 1e-6:
            print(f"  ⚠️  SEVERE VANISHING GRADIENT!")
        elif ratio < 0.1:
            print(f"  ⚠️  Moderate gradient vanishing")
        else:
            print(f"  ✅ Good gradient flow")
        print()
    
    # ──────────────────────────────────────
    # 6. VISUALIZATION
    # ──────────────────────────────────────
    print("Generating visualizations...\n")
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Gradient magnitudes (linear scale)
    ax1 = plt.subplot(2, 3, 1)
    for name, data in results.items():
        layer_indices = range(len(data['gradients']))
        plt.plot(layer_indices, data['gradients'], marker='o', 
                linewidth=2, label=name, markersize=4)
    
    plt.xlabel('Layer Index (0 = output, {} = input)'.format(num_layers-1), fontsize=11)
    plt.ylabel('Gradient Magnitude', fontsize=11)
    plt.title('Gradient Flow (Linear Scale)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    
    # Plot 2: Gradient magnitudes (log scale)
    ax2 = plt.subplot(2, 3, 2)
    for name, data in results.items():
        layer_indices = range(len(data['gradients']))
        # Filter out zeros for log scale
        grads = np.array(data['gradients'])
        grads[grads == 0] = 1e-10  # Replace zeros with small value
        plt.plot(layer_indices, grads, marker='o', 
                linewidth=2, label=name, markersize=4)
    
    plt.xlabel('Layer Index (0 = output, {} = input)'.format(num_layers-1), fontsize=11)
    plt.ylabel('Gradient Magnitude (log scale)', fontsize=11)
    plt.title('Gradient Flow (Log Scale)', fontsize=13, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, which='both')
    plt.gca().invert_xaxis()
    
    # Plot 3: Gradient ratio (compared to output layer)
    ax3 = plt.subplot(2, 3, 3)
    for name, data in results.items():
        gradients = np.array(data['gradients'])
        output_grad = gradients[0]
        ratios = gradients / (output_grad + 1e-10)  # Avoid division by zero
        layer_indices = range(len(ratios))
        plt.plot(layer_indices, ratios, marker='o', 
                linewidth=2, label=name, markersize=4)
    
    plt.xlabel('Layer Index', fontsize=11)
    plt.ylabel('Gradient Ratio (relative to output layer)', fontsize=11)
    plt.title('Relative Gradient Strength', fontsize=13, fontweight='bold')
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Output layer')
    plt.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Severe vanishing (1%)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.gca().invert_xaxis()
    
    # Plot 4: Activation means through layers
    ax4 = plt.subplot(2, 3, 4)
    for name, data in results.items():
        if data['activation_means']:
            layer_indices = range(len(data['activation_means']))
            plt.plot(layer_indices, data['activation_means'], marker='o',
                    linewidth=2, label=name, markersize=4)
    
    plt.xlabel('Layer Index', fontsize=11)
    plt.ylabel('Mean Activation Value', fontsize=11)
    plt.title('Activation Means Through Network', fontsize=13, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Activation standard deviations
    ax5 = plt.subplot(2, 3, 5)
    for name, data in results.items():
        if data['activation_stds']:
            layer_indices = range(len(data['activation_stds']))
            plt.plot(layer_indices, data['activation_stds'], marker='o',
                    linewidth=2, label=name, markersize=4)
    
    plt.xlabel('Layer Index', fontsize=11)
    plt.ylabel('Std Dev of Activations', fontsize=11)
    plt.title('Activation Variance Through Network', fontsize=13, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    table_data = []
    table_data.append(['Activation', 'Output Grad', 'Input Grad', 'Ratio', 'Status'])
    table_data.append(['─'*12, '─'*11, '─'*11, '─'*11, '─'*15])
    
    for name, data in results.items():
        output_grad = data['gradients'][0]
        input_grad = data['gradients'][-1]
        ratio = input_grad / output_grad if output_grad > 0 else 0
        
        if ratio < 1e-6:
            status = '🔴 Severe'
        elif ratio < 0.1:
            status = '🟡 Moderate'
        else:
            status = '🟢 Good'
        
        table_data.append([
            name,
            f'{output_grad:.6f}',
            f'{input_grad:.6f}',
            f'{ratio:.2e}',
            status
        ])
    
    table_text = '\n'.join(['  '.join(row) for row in table_data])
    
    summary = f"""
GRADIENT FLOW SUMMARY ({num_layers}-Layer Network)
{'='*65}

{table_text}

KEY OBSERVATIONS:
───────────────────────────────────────────────────────────
• Sigmoid/Tanh: Catastrophic gradient vanishing
  - Gradients approach zero in early layers
  - Training impossible for deep networks
  
• ReLU/LeakyReLU: Excellent gradient flow
  - Gradients remain strong throughout
  - Enables training of very deep networks
  
• GELU/SiLU: Good gradient flow
  - Slightly better than ReLU for some tasks
  - More computational cost
  
RECOMMENDATION:
───────────────────────────────────────────────────────────
For deep networks (>10 layers):
  ✅ USE: ReLU, LeakyReLU, GELU, SiLU
  ❌ AVOID: Sigmoid, Tanh (except in RNNs/LSTMs)
"""
    
    ax6.text(0.05, 0.95, summary, fontsize=9, family='monospace',
            verticalalignment='top', fontweight='normal')
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo2_gradient_flow.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to: demo2_gradient_flow.png")
    
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print("-" * 40)
        gradients = data['gradients']
        
        # Calculate gradient decay rate
        decay_rates = []
        for i in range(1, len(gradients)):
            if gradients[i-1] > 0:
                decay = gradients[i] / gradients[i-1]
                decay_rates.append(decay)
        
        avg_decay = np.mean(decay_rates) if decay_rates else 0
        
        print(f"  Average gradient decay per layer: {avg_decay:.4f}")
        print(f"  Total gradient reduction: {gradients[-1]/gradients[0]:.2e}x")
        
        if avg_decay < 0.5:
            print(f"  ⚠️  Each layer loses >{100*(1-avg_decay):.0f}% of gradient")
        elif avg_decay > 0.9:
            print(f"  ✅ Gradients well-preserved (only {100*(1-avg_decay):.0f}% loss per layer)")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    run_gradient_flow_experiment()
