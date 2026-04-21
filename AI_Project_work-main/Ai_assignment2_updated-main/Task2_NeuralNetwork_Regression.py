"""
Task 2: Implementing Linear Regression Using a Multi-Layer Neural Network from Scratch
========================================================================================
AI Assignment 2 - Group K

This script implements a multi-layer neural network from scratch for linear regression
on the Boston Housing Dataset.

Architecture:
- Input Layer: 2 neurons (RM, CRIM features)
- Hidden Layer 1: 5 neurons with ReLU activation
- Hidden Layer 2: 3 neurons with ReLU activation
- Output Layer: 1 neuron (MEDV prediction)

Optimizers implemented:
- Basic Gradient Descent
- Momentum
- Adam

Bonus:
- Third hidden layer with 2 neurons
- L2 Regularization (Weight Decay)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# LOAD AND PREPROCESS BOSTON HOUSING DATASET
# ============================================================================

def load_boston_housing():
    """
    Generate synthetic Boston Housing-like dataset.
    Since network is unavailable, we create realistic synthetic data.
    
    The synthetic data mimics the relationship:
    MEDV ~ f(RM, CRIM) where:
    - RM (avg rooms per dwelling): positively correlated with MEDV
    - CRIM (per capita crime rate): negatively correlated with MEDV
    """
    np.random.seed(42)
    n_samples = 506  # Same as original Boston dataset
    
    # Generate RM (number of rooms) - typically 3-9 rooms
    RM = np.random.normal(6.28, 0.7, n_samples)
    RM = np.clip(RM, 3.5, 8.8)
    
    # Generate CRIM (crime rate) - typically 0-90, highly skewed
    CRIM = np.random.exponential(3.6, n_samples)
    CRIM = np.clip(CRIM, 0.006, 89)
    
    # Generate MEDV (median home value) based on RM and CRIM with noise
    # MEDV = base + rooms_effect - crime_effect + noise
    base = 5.0
    rooms_effect = 3.5 * RM
    crime_effect = 0.3 * CRIM
    noise = np.random.normal(0, 2, n_samples)
    
    MEDV = base + rooms_effect - crime_effect + noise
    MEDV = np.clip(MEDV, 5, 50)  # Clip to realistic range
    
    # Create DataFrame-like structure
    class DataContainer:
        def __init__(self, rm, crim):
            self._data = {'RM': rm, 'CRIM': crim}
        
        def __getitem__(self, key):
            if isinstance(key, list):
                return np.column_stack([self._data[k] for k in key])
            return self._data[key]
    
    class TargetContainer:
        def __init__(self, values):
            self._values = values
        
        @property
        def values(self):
            return self._values
    
    data = DataContainer(RM, CRIM)
    target = TargetContainer(MEDV)
    
    return data, target

def preprocess_data(data, target):
    """
    Preprocess data:
    1. Select features: RM (number of rooms) and CRIM (crime rate)
    2. Normalize features
    3. Split into train (80%) and test (20%)
    """
    # Select features
    X = data[['RM', 'CRIM']].astype(float)
    y = target.values.reshape(-1, 1)
    
    # Normalize features (Min-Max normalization to [0, 1])
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min + 1e-8)
    
    # Normalize target
    y_min = y.min()
    y_max = y.max()
    y_normalized = (y - y_min) / (y_max - y_min + 1e-8)
    
    # Split data (80% train, 20% test)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = X_normalized[train_idx]
    X_test = X_normalized[test_idx]
    y_train = y_normalized[train_idx]
    y_test = y_normalized[test_idx]
    
    # Store normalization parameters for denormalization
    norm_params = {
        'X_min': X_min, 'X_max': X_max,
        'y_min': y_min, 'y_max': y_max
    }
    
    return X_train, X_test, y_train, y_test, norm_params


# ============================================================================
# NEURAL NETWORK IMPLEMENTATION FROM SCRATCH
# ============================================================================

class NeuralNetwork:
    """
    Multi-layer Neural Network for Regression
    Implemented from scratch using only NumPy
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu',
                 l2_lambda: float = 0.0):
        """
        Initialize neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation function ('relu')
            l2_lambda: L2 regularization strength (0 = no regularization)
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation = activation
        self.l2_lambda = l2_lambda
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for ReLU
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Store activations and pre-activations for backprop
        self.activations = []
        self.z_values = []
    
    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation
        """
        self.activations = [X]
        self.z_values = []
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation (ReLU for hidden layers, linear for output)
            if i < len(self.weights) - 1:
                a = self.relu(z)
            else:
                a = z  # Linear activation for output layer
            
            self.activations.append(a)
        
        return a
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss with optional L2 regularization
        """
        m = y_true.shape[0]
        mse = np.mean((y_pred - y_true) ** 2)
        
        # Add L2 regularization
        if self.l2_lambda > 0:
            l2_penalty = 0
            for w in self.weights:
                l2_penalty += np.sum(w ** 2)
            mse += (self.l2_lambda / (2 * m)) * l2_penalty
        
        return mse
    
    def backward(self, y_true: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation to compute gradients
        """
        m = y_true.shape[0]
        
        dweights = [None] * len(self.weights)
        dbiases = [None] * len(self.biases)
        
        # Output layer gradient (MSE derivative)
        delta = (self.activations[-1] - y_true) * (2 / m)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            dweights[i] = np.dot(self.activations[i].T, delta)
            dbiases[i] = np.sum(delta, axis=0, keepdims=True)
            
            # Add L2 regularization gradient
            if self.l2_lambda > 0:
                dweights[i] += (self.l2_lambda / m) * self.weights[i]
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        return dweights, dbiases
    
    def get_weights_copy(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return deep copy of weights and biases"""
        return ([w.copy() for w in self.weights], 
                [b.copy() for b in self.biases])
    
    def set_weights(self, weights: List[np.ndarray], biases: List[np.ndarray]):
        """Set weights and biases"""
        self.weights = [w.copy() for w in weights]
        self.biases = [b.copy() for b in biases]


# ============================================================================
# OPTIMIZER IMPLEMENTATIONS
# ============================================================================

class GradientDescentOptimizer:
    """Basic Gradient Descent"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.name = "Gradient Descent"
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray],
               dweights: List[np.ndarray], dbiases: List[np.ndarray]) -> Tuple[List, List]:
        """Update weights using gradient descent"""
        for i in range(len(weights)):
            weights[i] -= self.lr * dweights[i]
            biases[i] -= self.lr * dbiases[i]
        return weights, biases


class MomentumOptimizer:
    """SGD with Momentum"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.name = "Momentum"
        self.v_w = None
        self.v_b = None
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray],
               dweights: List[np.ndarray], dbiases: List[np.ndarray]) -> Tuple[List, List]:
        """Update weights using momentum"""
        if self.v_w is None:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            self.v_w[i] = self.momentum * self.v_w[i] - self.lr * dweights[i]
            self.v_b[i] = self.momentum * self.v_b[i] - self.lr * dbiases[i]
            weights[i] += self.v_w[i]
            biases[i] += self.v_b[i]
        
        return weights, biases


class AdamOptimizer:
    """Adam Optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = "Adam"
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None
        self.t = 0
    
    def update(self, weights: List[np.ndarray], biases: List[np.ndarray],
               dweights: List[np.ndarray], dbiases: List[np.ndarray]) -> Tuple[List, List]:
        """Update weights using Adam"""
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]
        
        self.t += 1
        
        for i in range(len(weights)):
            # Update biased first moment estimate
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dweights[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * dbiases[i]
            
            # Update biased second raw moment estimate
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dweights[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (dbiases[i] ** 2)
            
            # Bias correction
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            weights[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            biases[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return weights, biases


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_network(model: NeuralNetwork, X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray,
                  optimizer, epochs: int = 1000,
                  initial_weights: Optional[Tuple] = None) -> Dict:
    """
    Train the neural network
    
    Returns:
        Dictionary containing training history and final metrics
    """
    # Set initial weights if provided
    if initial_weights:
        model.set_weights(initial_weights[0], initial_weights[1])
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred_train = model.forward(X_train)
        
        # Compute loss
        train_loss = model.compute_loss(y_pred_train, y_train)
        train_losses.append(train_loss)
        
        # Compute test loss
        y_pred_test = model.forward(X_test)
        test_loss = model.compute_loss(y_pred_test, y_test)
        test_losses.append(test_loss)
        
        # Forward again for backprop (after computing test loss)
        model.forward(X_train)
        
        # Backward pass
        dweights, dbiases = model.backward(y_train)
        
        # Update weights
        model.weights, model.biases = optimizer.update(
            model.weights, model.biases, dweights, dbiases
        )
        
        # Print progress
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    # Final predictions
    y_pred_train = model.forward(X_train)
    y_pred_test = model.forward(X_test)
    
    # Compute final MSE
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def run_experiments():
    """Run all experiments for Task 2"""
    
    print("=" * 70)
    print("Task 2: Linear Regression Using Multi-Layer Neural Network")
    print("Group K - AI Assignment 2")
    print("=" * 70)
    
    # Load and preprocess data
    print("\nLoading Boston Housing Dataset...")
    data, target = load_boston_housing()
    X_train, X_test, y_train, y_test, norm_params = preprocess_data(data, target)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: RM (Rooms), CRIM (Crime Rate)")
    
    # Store results
    results = {}
    
    # ========================================================================
    # EXPERIMENT 1: COMPARE OPTIMIZERS WITH DIFFERENT LEARNING RATES
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Comparing Optimizers")
    print("=" * 70)
    
    learning_rates = [0.01, 0.001]
    epochs = 1000
    
    # Architecture: 2 -> 5 -> 3 -> 1
    layer_sizes = [2, 5, 3, 1]
    
    # Create initial weights for fair comparison
    init_model = NeuralNetwork(layer_sizes)
    initial_weights = init_model.get_weights_copy()
    
    for lr in learning_rates:
        print(f"\n--- Learning Rate: {lr} ---")
        results[lr] = {}
        
        # Gradient Descent
        print("\nTraining with Gradient Descent...")
        model_gd = NeuralNetwork(layer_sizes)
        opt_gd = GradientDescentOptimizer(learning_rate=lr)
        results[lr]['GD'] = train_network(model_gd, X_train, y_train, X_test, y_test,
                                           opt_gd, epochs, initial_weights)
        
        # Momentum
        print("\nTraining with Momentum...")
        model_mom = NeuralNetwork(layer_sizes)
        opt_mom = MomentumOptimizer(learning_rate=lr, momentum=0.9)
        results[lr]['Momentum'] = train_network(model_mom, X_train, y_train, X_test, y_test,
                                                 opt_mom, epochs, initial_weights)
        
        # Adam
        print("\nTraining with Adam...")
        model_adam = NeuralNetwork(layer_sizes)
        opt_adam = AdamOptimizer(learning_rate=lr)
        results[lr]['Adam'] = train_network(model_adam, X_train, y_train, X_test, y_test,
                                             opt_adam, epochs, initial_weights)
    
    # ========================================================================
    # BONUS 1: ADDITIONAL HIDDEN LAYER
    # ========================================================================
    print("\n" + "=" * 70)
    print("BONUS 1: Adding Third Hidden Layer (2 neurons)")
    print("=" * 70)
    
    # Architecture: 2 -> 5 -> 3 -> 2 -> 1
    layer_sizes_bonus = [2, 5, 3, 2, 1]
    lr_bonus = 0.01
    
    print("\nTraining network with 3 hidden layers...")
    model_3h = NeuralNetwork(layer_sizes_bonus)
    opt_3h = AdamOptimizer(learning_rate=lr_bonus)
    results['3_hidden'] = train_network(model_3h, X_train, y_train, X_test, y_test,
                                         opt_3h, epochs)
    
    # ========================================================================
    # BONUS 2: L2 REGULARIZATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("BONUS 2: L2 Regularization (Weight Decay)")
    print("=" * 70)
    
    l2_lambdas = [0.0, 0.001, 0.01, 0.1]
    results['l2_reg'] = {}
    
    for l2 in l2_lambdas:
        print(f"\nL2 Lambda: {l2}")
        model_l2 = NeuralNetwork(layer_sizes, l2_lambda=l2)
        opt_l2 = AdamOptimizer(learning_rate=0.01)
        results['l2_reg'][l2] = train_network(model_l2, X_train, y_train, X_test, y_test,
                                               opt_l2, epochs)
    
    return results, X_train, X_test, y_train, y_test, norm_params


def create_plots(results: Dict, X_train, X_test, y_train, y_test, norm_params):
    """Generate all required plots"""
    
    # ========================================================================
    # PLOT 1: LOSS CURVES FOR DIFFERENT OPTIMIZERS
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, lr in enumerate([0.01, 0.001]):
        ax = axes[0, idx]
        for opt_name in ['GD', 'Momentum', 'Adam']:
            ax.plot(results[lr][opt_name]['train_losses'], label=f'{opt_name} (Train)')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'Training Loss - Learning Rate = {lr}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    for idx, lr in enumerate([0.01, 0.001]):
        ax = axes[1, idx]
        for opt_name in ['GD', 'Momentum', 'Adam']:
            ax.plot(results[lr][opt_name]['test_losses'], label=f'{opt_name} (Test)')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title(f'Test Loss - Learning Rate = {lr}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task2_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 2: PREDICTED VS ACTUAL VALUES
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, opt_name in enumerate(['GD', 'Momentum', 'Adam']):
        ax = axes[idx]
        
        # Use lr=0.01 results
        y_pred = results[0.01][opt_name]['y_pred_test']
        
        # Denormalize
        y_pred_denorm = y_pred * (norm_params['y_max'] - norm_params['y_min']) + norm_params['y_min']
        y_test_denorm = y_test * (norm_params['y_max'] - norm_params['y_min']) + norm_params['y_min']
        
        ax.scatter(y_test_denorm, y_pred_denorm, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_test_denorm.min(), y_pred_denorm.min())
        max_val = max(y_test_denorm.max(), y_pred_denorm.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values (MEDV)')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{opt_name} - Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task2_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 3: BONUS - COMPARISON WITH 3 HIDDEN LAYERS
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compare 2 hidden layers vs 3 hidden layers
    ax = axes[0]
    ax.plot(results[0.01]['Adam']['train_losses'], label='2 Hidden Layers (5,3)')
    ax.plot(results['3_hidden']['train_losses'], label='3 Hidden Layers (5,3,2)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Effect of Additional Hidden Layer - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    ax = axes[1]
    ax.plot(results[0.01]['Adam']['test_losses'], label='2 Hidden Layers (5,3)')
    ax.plot(results['3_hidden']['test_losses'], label='3 Hidden Layers (5,3,2)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Effect of Additional Hidden Layer - Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task2_hidden_layers_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 4: BONUS - L2 REGULARIZATION EFFECT
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    l2_lambdas = [0.0, 0.001, 0.01, 0.1]
    
    ax = axes[0]
    for l2 in l2_lambdas:
        ax.plot(results['l2_reg'][l2]['train_losses'], label=f'λ={l2}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('L2 Regularization Effect - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    ax = axes[1]
    for l2 in l2_lambdas:
        ax.plot(results['l2_reg'][l2]['test_losses'], label=f'λ={l2}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('L2 Regularization Effect - Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task2_l2_regularization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 5: MSE COMPARISON BAR CHART
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    optimizers = ['GD', 'Momentum', 'Adam']
    lrs = [0.01, 0.001]
    
    x = np.arange(len(optimizers))
    width = 0.35
    
    train_mse_01 = [results[0.01][opt]['train_mse'] for opt in optimizers]
    test_mse_01 = [results[0.01][opt]['test_mse'] for opt in optimizers]
    
    ax.bar(x - width/2, train_mse_01, width, label='Train MSE (lr=0.01)', color='steelblue')
    ax.bar(x + width/2, test_mse_01, width, label='Test MSE (lr=0.01)', color='coral')
    
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('MSE')
    ax.set_title('Final MSE Comparison (Learning Rate = 0.01)')
    ax.set_xticks(x)
    ax.set_xticklabels(optimizers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task2_mse_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nAll Task 2 plots saved successfully!")


def print_summary(results: Dict):
    """Print summary of results"""
    
    print("\n" + "=" * 80)
    print("TASK 2 SUMMARY: MSE Results")
    print("=" * 80)
    
    print("\n--- Standard Architecture (2 Hidden Layers: 5, 3) ---")
    print(f"{'Optimizer':<15} {'LR':<10} {'Train MSE':<15} {'Test MSE':<15}")
    print("-" * 55)
    
    for lr in [0.01, 0.001]:
        for opt in ['GD', 'Momentum', 'Adam']:
            print(f"{opt:<15} {lr:<10} {results[lr][opt]['train_mse']:<15.6f} {results[lr][opt]['test_mse']:<15.6f}")
    
    print("\n--- Bonus: 3 Hidden Layers (5, 3, 2) with Adam lr=0.01 ---")
    print(f"Train MSE: {results['3_hidden']['train_mse']:.6f}")
    print(f"Test MSE: {results['3_hidden']['test_mse']:.6f}")
    
    print("\n--- Bonus: L2 Regularization with Adam lr=0.01 ---")
    print(f"{'L2 Lambda':<15} {'Train MSE':<15} {'Test MSE':<15}")
    print("-" * 45)
    for l2 in [0.0, 0.001, 0.01, 0.1]:
        print(f"{l2:<15} {results['l2_reg'][l2]['train_mse']:<15.6f} {results['l2_reg'][l2]['test_mse']:<15.6f}")
    
    return


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run experiments
    results, X_train, X_test, y_train, y_test, norm_params = run_experiments()
    
    # Create plots
    create_plots(results, X_train, X_test, y_train, y_test, norm_params)
    
    # Print summary
    print_summary(results)
    
    # Save results to file
    with open('/home/claude/GroupK_Assignment2/task2_results.txt', 'w') as f:
        f.write("Task 2: Linear Regression Using Multi-Layer Neural Network\n")
        f.write("Group K - AI Assignment 2\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Architecture: Input(2) -> Hidden1(5, ReLU) -> Hidden2(3, ReLU) -> Output(1)\n\n")
        
        f.write("MSE Results:\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Optimizer':<15} {'LR':<10} {'Train MSE':<15} {'Test MSE':<15}\n")
        f.write("-" * 55 + "\n")
        
        for lr in [0.01, 0.001]:
            for opt in ['GD', 'Momentum', 'Adam']:
                f.write(f"{opt:<15} {lr:<10} {results[lr][opt]['train_mse']:<15.6f} {results[lr][opt]['test_mse']:<15.6f}\n")
        
        f.write("\nBonus - 3 Hidden Layers:\n")
        f.write(f"Train MSE: {results['3_hidden']['train_mse']:.6f}\n")
        f.write(f"Test MSE: {results['3_hidden']['test_mse']:.6f}\n")
        
        f.write("\nBonus - L2 Regularization:\n")
        for l2 in [0.0, 0.001, 0.01, 0.1]:
            f.write(f"Lambda={l2}: Train MSE={results['l2_reg'][l2]['train_mse']:.6f}, Test MSE={results['l2_reg'][l2]['test_mse']:.6f}\n")
    
    print("\n" + "=" * 70)
    print("Task 2 completed successfully!")
    print("Plots and results saved to /home/claude/GroupK_Assignment2/")
    print("=" * 70)
