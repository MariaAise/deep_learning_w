"""
DEMO 1: Visualization - Network With/Without Activations on 2D Classification

This demo shows why activation functions are necessary by comparing:
1. Linear model (no activations) - can only create straight decision boundaries
2. Non-linear model (with activations) - can create complex decision boundaries
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ──────────────────────────────────────
# 1. GENERATE SYNTHETIC DATA
# ──────────────────────────────────────
def generate_spiral_data(n_samples=500, noise=0.1):
    """Generate spiral dataset that's not linearly separable"""
    n_per_class = n_samples // 2
    theta = np.linspace(0, 4 * np.pi, n_per_class)
    
    # Class 0: inner spiral
    r0 = theta / (2 * np.pi)
    x0 = r0 * np.cos(theta) + noise * np.random.randn(n_per_class)
    y0 = r0 * np.sin(theta) + noise * np.random.randn(n_per_class)
    
    # Class 1: outer spiral
    r1 = (theta + np.pi) / (2 * np.pi)
    x1 = r1 * np.cos(theta) + noise * np.random.randn(n_per_class)
    y1 = r1 * np.sin(theta) + noise * np.random.randn(n_per_class)
    
    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X.astype(np.float32), y.astype(np.int64)

def generate_circle_data(n_samples=500, noise=0.05):
    """Generate circular dataset"""
    n_per_class = n_samples // 2
    theta = np.random.uniform(0, 2 * np.pi, n_per_class)
    
    # Class 0: inner circle
    r0 = np.random.uniform(0, 1, n_per_class)
    x0 = r0 * np.cos(theta) + noise * np.random.randn(n_per_class)
    y0 = r0 * np.sin(theta) + noise * np.random.randn(n_per_class)
    
    # Class 1: outer ring
    r1 = np.random.uniform(1.5, 2.5, n_per_class)
    x1 = r1 * np.cos(theta) + noise * np.random.randn(n_per_class)
    y1 = r1 * np.sin(theta) + noise * np.random.randn(n_per_class)
    
    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X.astype(np.float32), y.astype(np.int64)

# ──────────────────────────────────────
# 2. MODELS
# ──────────────────────────────────────
class LinearModel(nn.Module):
    """Model WITHOUT activation functions"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            # NO ACTIVATION!
            nn.Linear(64, 32),
            # NO ACTIVATION!
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

class NonLinearModel(nn.Module):
    """Model WITH activation functions"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),  # Activation!
            nn.Linear(64, 32),
            nn.ReLU(),  # Activation!
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

# ──────────────────────────────────────
# 3. TRAINING FUNCTION
# ──────────────────────────────────────
def train_model(model, X, y, epochs=1000, lr=0.01):
    """Train a model and return loss history"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).float().mean().item() * 100
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Acc: {accuracy:.2f}%")
    
    return losses

# ──────────────────────────────────────
# 4. VISUALIZATION FUNCTION
# ──────────────────────────────────────
def plot_decision_boundary(model, X, y, title):
    """Plot the decision boundary of a model"""
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_input = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    model.eval()
    with torch.no_grad():
        Z = model(mesh_input)
        Z = torch.softmax(Z, dim=1)[:, 1].numpy()
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Decision boundary
    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdYlBu', alpha=0.6)
    plt.colorbar(label='P(Class 1)')
    
    # Data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                         edgecolors='black', linewidth=1.5, s=50)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    return plt

# ──────────────────────────────────────
# 5. RUN COMPARISON
# ──────────────────────────────────────
def run_comparison():
    """Compare linear vs non-linear models on non-linear dataset"""
    print("="*60)
    print("DEMO 1: Necessity of Activation Functions")
    print("="*60)
    
    # Generate data
    print("\nGenerating spiral dataset...")
    X, y = generate_spiral_data(n_samples=500, noise=0.15)
    
    # Train linear model (no activations)
    print("\n" + "-"*60)
    print("Training LINEAR model (no activation functions)...")
    print("-"*60)
    linear_model = LinearModel()
    linear_losses = train_model(linear_model, X, y, epochs=1000, lr=0.01)
    
    # Train non-linear model (with activations)
    print("\n" + "-"*60)
    print("Training NON-LINEAR model (with ReLU activations)...")
    print("-"*60)
    nonlinear_model = NonLinearModel()
    nonlinear_losses = train_model(nonlinear_model, X, y, epochs=1000, lr=0.01)
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Subplot 1: Linear model decision boundary
    plt.subplot(2, 3, 1)
    plot_decision_boundary(linear_model, X, y, 
                          "Linear Model (NO Activation)\n❌ Cannot Learn Non-Linear Boundary")
    
    # Subplot 2: Non-linear model decision boundary
    plt.subplot(2, 3, 2)
    plot_decision_boundary(nonlinear_model, X, y,
                          "Non-Linear Model (WITH ReLU)\n✅ Learns Complex Boundary")
    
    # Subplot 3: Loss comparison
    plt.subplot(2, 3, 3)
    plt.plot(linear_losses, label='Linear Model', linewidth=2, alpha=0.8)
    plt.plot(nonlinear_losses, label='Non-Linear Model', linewidth=2, alpha=0.8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Test on circle dataset too
    print("\nGenerating circular dataset...")
    X_circle, y_circle = generate_circle_data(n_samples=500, noise=0.1)
    
    # Train models on circle data
    print("\nTraining models on circular data...")
    linear_model_circle = LinearModel()
    train_model(linear_model_circle, X_circle, y_circle, epochs=500, lr=0.01)
    
    nonlinear_model_circle = NonLinearModel()
    train_model(nonlinear_model_circle, X_circle, y_circle, epochs=500, lr=0.01)
    
    # Subplot 4: Linear model on circles
    plt.subplot(2, 3, 4)
    plot_decision_boundary(linear_model_circle, X_circle, y_circle,
                          "Linear Model on Circles\n❌ Straight Line Only")
    
    # Subplot 5: Non-linear model on circles
    plt.subplot(2, 3, 5)
    plot_decision_boundary(nonlinear_model_circle, X_circle, y_circle,
                          "Non-Linear Model on Circles\n✅ Circular Boundary")
    
    # Subplot 6: Mathematical explanation
    plt.subplot(2, 3, 6)
    plt.axis('off')
    explanation = """
    MATHEMATICAL PROOF
    ══════════════════════════════════════
    
    WITHOUT Activations:
    ──────────────────────────────────────
    Layer 1: h₁ = W₁x + b₁
    Layer 2: h₂ = W₂h₁ + b₂
    Layer 3: y = W₃h₂ + b₃
    
    Substituting:
    y = W₃(W₂(W₁x + b₁) + b₂) + b₃
    y = W₃W₂W₁x + W₃W₂b₁ + W₃b₂ + b₃
    y = Wx + b  (collapsed to single layer!)
    
    Result: Can only learn LINEAR boundary
    ══════════════════════════════════════
    
    WITH Activations (ReLU):
    ──────────────────────────────────────
    Layer 1: h₁ = ReLU(W₁x + b₁)
    Layer 2: h₂ = ReLU(W₂h₁ + b₂)
    Layer 3: y = W₃h₂ + b₃
    
    Cannot be simplified!
    Each ReLU creates piecewise linear regions
    Multiple layers → complex boundaries
    
    Result: Can learn CIRCULAR, SPIRAL,
            and any CONTINUOUS function!
    ══════════════════════════════════════
    """
    plt.text(0.1, 0.5, explanation, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo1_activation_necessity.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved to: demo1_activation_necessity.png")
    
    plt.show()
    
    # Final accuracy comparison
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    # Spiral dataset
    with torch.no_grad():
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
        
        linear_out = linear_model(X_tensor)
        linear_acc = ((linear_out.argmax(1) == y_tensor).float().mean() * 100).item()
        
        nonlinear_out = nonlinear_model(X_tensor)
        nonlinear_acc = ((nonlinear_out.argmax(1) == y_tensor).float().mean() * 100).item()
    
    print(f"\nSpiral Dataset:")
    print(f"  Linear Model (no activation):     {linear_acc:.2f}% accuracy")
    print(f"  Non-Linear Model (with ReLU):     {nonlinear_acc:.2f}% accuracy")
    print(f"  Improvement:                      +{nonlinear_acc - linear_acc:.2f}%")
    
    # Circle dataset
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_circle)
        y_tensor = torch.from_numpy(y_circle)
        
        linear_acc_circle = ((linear_model_circle(X_tensor).argmax(1) == y_tensor).float().mean() * 100).item()
        nonlinear_acc_circle = ((nonlinear_model_circle(X_tensor).argmax(1) == y_tensor).float().mean() * 100).item()
    
    print(f"\nCircular Dataset:")
    print(f"  Linear Model (no activation):     {linear_acc_circle:.2f}% accuracy")
    print(f"  Non-Linear Model (with ReLU):     {nonlinear_acc_circle:.2f}% accuracy")
    print(f"  Improvement:                      +{nonlinear_acc_circle - linear_acc_circle:.2f}%")
    
    print("\n" + "="*60)
    print("KEY TAKEAWAY: Activation functions are ESSENTIAL for")
    print("learning non-linear patterns. Without them, deep networks")
    print("collapse to simple linear models, regardless of depth!")
    print("="*60)

if __name__ == "__main__":
    run_comparison()
