"""
Task 3: Multi-class Classification using Fully Connected Neural Network
=========================================================================
AI Assignment 2 - Group K

This script implements a Fully Connected Neural Network from scratch for
multi-class classification on:
1. Linearly separable dataset (3 classes, 2D)
2. Non-linearly separable dataset (3 classes, 2D)

Architecture:
- Dataset 1: 1 hidden layer
- Dataset 2: 2 hidden layers

Loss function: Squared Error
Optimizer: Stochastic Gradient Descent (SGD)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_linearly_separable_data(n_per_class: int = 500):
    """Generate 3-class, 2D linearly separable data."""
    np.random.seed(42)
    X0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-2, -2])
    y0 = np.zeros(n_per_class, dtype=int)
    X1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([0, 2])
    y1 = np.ones(n_per_class, dtype=int)
    X2 = np.random.randn(n_per_class, 2) * 0.5 + np.array([2, -2])
    y2 = np.full(n_per_class, 2, dtype=int)
    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])
    return X, y

def generate_nonlinearly_separable_data(n_per_class: int = 500):
    """Generate 3-class, 2D non-linearly separable data (concentric circles)."""
    np.random.seed(42)
    theta0 = np.random.uniform(0, 2*np.pi, n_per_class)
    r0 = np.random.uniform(0, 1, n_per_class)
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    y0 = np.zeros(n_per_class, dtype=int)
    
    theta1 = np.random.uniform(0, 2*np.pi, n_per_class)
    r1 = np.random.uniform(1.5, 2.5, n_per_class)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.ones(n_per_class, dtype=int)
    
    theta2 = np.random.uniform(0, 2*np.pi, n_per_class)
    r2 = np.random.uniform(3, 4, n_per_class)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = np.full(n_per_class, 2, dtype=int)
    
    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])
    return X, y

def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
    """Split data into train/val/test with stratification."""
    n_classes = len(np.unique(y))
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        n_val = int(len(idx) * val_ratio)
        
        X_train.append(X[idx[:n_train]])
        y_train.append(y[idx[:n_train]])
        X_val.append(X[idx[n_train:n_train + n_val]])
        y_val.append(y[idx[n_train:n_train + n_val]])
        X_test.append(X[idx[n_train + n_val:]])
        y_test.append(y[idx[n_train + n_val:]])
    
    return {
        'X_train': np.vstack(X_train), 'y_train': np.hstack(y_train),
        'X_val': np.vstack(X_val), 'y_val': np.hstack(y_val),
        'X_test': np.vstack(X_test), 'y_test': np.hstack(y_test)
    }

def normalize_data(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std

def one_hot_encode(y, n_classes):
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# ============================================================================
# NEURAL NETWORK
# ============================================================================

class FCNNClassifier:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        self.activations = []
        self.z_values = []
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
        return a
    
    def compute_loss(self, y_pred, y_true):
        return 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1))
    
    def backward_sgd(self, y_true, learning_rate):
        delta = (self.activations[-1] - y_true) * self.sigmoid_derivative(self.activations[-1])
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
    
    def train_sgd(self, X_train, y_train, X_val, y_val, learning_rate=0.1, max_epochs=1000, tol=1e-4):
        n_samples = X_train.shape[0]
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        for epoch in range(max_epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            for idx in indices:
                x_i = X_train[idx:idx+1]
                y_i = y_train[idx:idx+1]
                y_pred = self.forward(x_i)
                self.backward_sgd(y_i, learning_rate)
                epoch_loss += self.compute_loss(y_pred, y_i)
            
            avg_train_loss = epoch_loss / n_samples
            train_losses.append(avg_train_loss)
            
            y_val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val_pred, y_val)
            val_losses.append(val_loss)
            
            train_pred = self.predict(X_train)
            train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))
            train_accuracies.append(train_acc)
            
            val_pred = self.predict(X_val)
            val_acc = np.mean(val_pred == np.argmax(y_val, axis=1))
            val_accuracies.append(val_acc)
            
            if len(train_losses) > 1 and abs(train_losses[-1] - train_losses[-2]) < tol:
                print(f"Converged at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_train_loss:.6f}, Val Acc: {val_acc:.4f}")
        
        return {'train_losses': train_losses, 'val_losses': val_losses,
                'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def get_hidden_outputs(self, X):
        self.forward(X)
        return self.activations[1:]

# ============================================================================
# SINGLE NEURON (PERCEPTRON)
# ============================================================================

class SingleNeuron:
    def __init__(self, n_inputs, n_outputs):
        self.weights = np.random.randn(n_inputs, n_outputs) * 0.1
        self.bias = np.zeros((1, n_outputs))
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def train(self, X_train, y_train, X_val, y_val, lr=0.1, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            error = (y_pred - y_train) * y_pred * (1 - y_pred)
            dw = np.dot(X_train.T, error) / len(X_train)
            db = np.mean(error, axis=0, keepdims=True)
            self.weights -= lr * dw
            self.bias -= lr * db
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# ============================================================================
# EVALUATION & PLOTTING
# ============================================================================

def compute_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def plot_decision_regions(model, X, y, title, filename, mean, std):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_normalized = (grid_points - mean) / std
    Z = model.predict(grid_normalized).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    colors = ['red', 'green', 'blue']
    X_denorm = X * std + mean
    for i in range(3):
        idx = y == i
        plt.scatter(X_denorm[idx, 0], X_denorm[idx, 1], c=colors[i], 
                   label=f'Class {i}', edgecolors='black', s=30, alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    n_classes = cm.shape[0]
    plt.xticks(range(n_classes), [f'Class {i}' for i in range(n_classes)])
    plt.yticks(range(n_classes), [f'Class {i}' for i in range(n_classes)])
    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def cross_validate_architecture(X_train, y_train, X_val, y_val, architectures, learning_rate=0.1, max_epochs=500):
    results = {}
    for arch in architectures:
        print(f"\nTesting architecture: {arch}")
        model = FCNNClassifier(arch)
        history = model.train_sgd(X_train, y_train, X_val, y_val, learning_rate=learning_rate, max_epochs=max_epochs)
        val_pred = model.predict(X_val)
        val_true = np.argmax(y_val, axis=1)
        val_acc = compute_accuracy(val_true, val_pred)
        cm = compute_confusion_matrix(val_true, val_pred, n_classes=arch[-1])
        results[tuple(arch)] = {'model': model, 'history': history, 'val_accuracy': val_acc, 'confusion_matrix': cm}
        print(f"Validation Accuracy: {val_acc:.4f}")
    return results

# ============================================================================
# MAIN
# ============================================================================

def run_task3():
    print("=" * 70)
    print("Task 3: Multi-class Classification using FCNN")
    print("Group K - AI Assignment 2")
    print("=" * 70)
    
    results = {}
    
    # Dataset 1: Linearly Separable
    print("\n" + "=" * 70)
    print("DATASET 1: Linearly Separable (3 classes, 2D)")
    print("=" * 70)
    
    X1, y1 = generate_linearly_separable_data(500)
    data1 = split_data(X1, y1)
    X1_train_norm, X1_val_norm, X1_test_norm, mean1, std1 = normalize_data(
        data1['X_train'], data1['X_val'], data1['X_test'])
    y1_train_oh = one_hot_encode(data1['y_train'], 3)
    y1_val_oh = one_hot_encode(data1['y_val'], 3)
    
    print(f"Train: {len(X1_train_norm)}, Val: {len(X1_val_norm)}, Test: {len(X1_test_norm)}")
    
    architectures_1 = [[2, 3, 3], [2, 5, 3], [2, 10, 3], [2, 15, 3]]
    results_1 = cross_validate_architecture(X1_train_norm, y1_train_oh, X1_val_norm, y1_val_oh,
                                           architectures_1, learning_rate=0.5, max_epochs=500)
    
    best_arch_1 = max(results_1.keys(), key=lambda k: results_1[k]['val_accuracy'])
    best_model_1 = results_1[best_arch_1]['model']
    test_pred_1 = best_model_1.predict(X1_test_norm)
    test_acc_1 = compute_accuracy(data1['y_test'], test_pred_1)
    test_cm_1 = compute_confusion_matrix(data1['y_test'], test_pred_1, 3)
    
    print(f"\nBest Architecture: {list(best_arch_1)}, Test Accuracy: {test_acc_1:.4f}")
    
    results['dataset1'] = {
        'results': results_1, 'best_arch': best_arch_1, 'test_accuracy': test_acc_1,
        'test_cm': test_cm_1, 'best_model': best_model_1, 'mean': mean1, 'std': std1,
        'data': data1, 'X_train_norm': X1_train_norm, 'y_train_oh': y1_train_oh
    }
    
    # Dataset 2: Non-Linearly Separable
    print("\n" + "=" * 70)
    print("DATASET 2: Non-Linearly Separable (3 classes, 2D)")
    print("=" * 70)
    
    X2, y2 = generate_nonlinearly_separable_data(500)
    data2 = split_data(X2, y2)
    X2_train_norm, X2_val_norm, X2_test_norm, mean2, std2 = normalize_data(
        data2['X_train'], data2['X_val'], data2['X_test'])
    y2_train_oh = one_hot_encode(data2['y_train'], 3)
    y2_val_oh = one_hot_encode(data2['y_val'], 3)
    
    print(f"Train: {len(X2_train_norm)}, Val: {len(X2_val_norm)}, Test: {len(X2_test_norm)}")
    
    architectures_2 = [[2, 5, 3, 3], [2, 10, 5, 3], [2, 15, 8, 3], [2, 20, 10, 3]]
    results_2 = cross_validate_architecture(X2_train_norm, y2_train_oh, X2_val_norm, y2_val_oh,
                                           architectures_2, learning_rate=0.5, max_epochs=500)
    
    best_arch_2 = max(results_2.keys(), key=lambda k: results_2[k]['val_accuracy'])
    best_model_2 = results_2[best_arch_2]['model']
    test_pred_2 = best_model_2.predict(X2_test_norm)
    test_acc_2 = compute_accuracy(data2['y_test'], test_pred_2)
    test_cm_2 = compute_confusion_matrix(data2['y_test'], test_pred_2, 3)
    
    print(f"\nBest Architecture: {list(best_arch_2)}, Test Accuracy: {test_acc_2:.4f}")
    
    results['dataset2'] = {
        'results': results_2, 'best_arch': best_arch_2, 'test_accuracy': test_acc_2,
        'test_cm': test_cm_2, 'best_model': best_model_2, 'mean': mean2, 'std': std2,
        'data': data2, 'X_train_norm': X2_train_norm, 'y_train_oh': y2_train_oh
    }
    
    # Single Neuron Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH SINGLE NEURON MODEL")
    print("=" * 70)
    
    single_1 = SingleNeuron(2, 3)
    single_1.train(X1_train_norm, y1_train_oh, X1_val_norm, y1_val_oh, lr=0.5, epochs=500)
    single_acc_1 = compute_accuracy(data1['y_test'], single_1.predict(X1_test_norm))
    
    single_2 = SingleNeuron(2, 3)
    single_2.train(X2_train_norm, y2_train_oh, X2_val_norm, y2_val_oh, lr=0.5, epochs=500)
    single_acc_2 = compute_accuracy(data2['y_test'], single_2.predict(X2_test_norm))
    
    print(f"Dataset 1: FCNN={test_acc_1:.4f}, Single Neuron={single_acc_1:.4f}")
    print(f"Dataset 2: FCNN={test_acc_2:.4f}, Single Neuron={single_acc_2:.4f}")
    
    results['single_neuron'] = {'dataset1_acc': single_acc_1, 'dataset2_acc': single_acc_2}
    
    return results

def create_all_plots(results):
    # Error vs Epochs
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (ds, ax) in enumerate(zip(['dataset1', 'dataset2'], axes)):
        history = results[ds]['results'][results[ds]['best_arch']]['history']
        ax.plot(history['train_losses'], label='Train')
        ax.plot(history['val_losses'], label='Validation')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Error')
        ax.set_title(f"Dataset {idx+1} - Best Arch: {list(results[ds]['best_arch'])}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task3_error_vs_epochs.png', dpi=150)
    plt.close()
    
    # Decision Regions
    for ds in ['dataset1', 'dataset2']:
        plot_decision_regions(
            results[ds]['best_model'], results[ds]['X_train_norm'], results[ds]['data']['y_train'],
            f"{ds} Decision Regions", f'/home/claude/GroupK_Assignment2/task3_decision_{ds}.png',
            results[ds]['mean'], results[ds]['std'])
    
    # Confusion Matrices
    for ds in ['dataset1', 'dataset2']:
        plot_confusion_matrix(results[ds]['test_cm'], f"{ds} Test Confusion Matrix",
                            f'/home/claude/GroupK_Assignment2/task3_cm_{ds}.png')
    
    # FCNN vs Single Neuron
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35
    fcnn = [results['dataset1']['test_accuracy'], results['dataset2']['test_accuracy']]
    single = [results['single_neuron']['dataset1_acc'], results['single_neuron']['dataset2_acc']]
    ax.bar(x - width/2, fcnn, width, label='FCNN', color='steelblue')
    ax.bar(x + width/2, single, width, label='Single Neuron', color='coral')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('FCNN vs Single Neuron')
    ax.set_xticks(x)
    ax.set_xticklabels(['Dataset 1 (Linear)', 'Dataset 2 (Non-Linear)'])
    ax.legend()
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/task3_comparison.png', dpi=150)
    plt.close()
    
    print("All Task 3 plots saved!")

if __name__ == "__main__":
    results = run_task3()
    create_all_plots(results)
    
    with open('/home/claude/GroupK_Assignment2/task3_results.txt', 'w') as f:
        f.write("Task 3: Multi-class Classification using FCNN\n")
        f.write("=" * 70 + "\n\n")
        for ds in ['dataset1', 'dataset2']:
            f.write(f"{ds.upper()}\n")
            f.write(f"Best Architecture: {list(results[ds]['best_arch'])}\n")
            f.write(f"Test Accuracy: {results[ds]['test_accuracy']:.4f}\n")
            f.write(f"Confusion Matrix:\n{results[ds]['test_cm']}\n\n")
        f.write("Single Neuron Comparison:\n")
        f.write(f"Dataset 1: {results['single_neuron']['dataset1_acc']:.4f}\n")
        f.write(f"Dataset 2: {results['single_neuron']['dataset2_acc']:.4f}\n")
    
    print("\nTask 3 completed!")
