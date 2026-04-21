"""
Task 1: Optimizer Performance on Non-Convex Functions
======================================================
AI Assignment 2 - Group K

This script implements various optimization algorithms from scratch and tests them
on two non-convex functions:
1. Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
2. f(x) = sin(1/x) with f(0) = 0

Optimizers implemented:
- Gradient Descent
- Stochastic Gradient Descent with Momentum
- Adam
- RMSprop
- Adagrad
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Global minimum at (1, 1) with f(1,1) = 0
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of Rosenbrock function
    df/dx = -2(1-x) - 400x(y-x^2)
    df/dy = 200(y-x^2)
    """
    grad = np.zeros(2)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def sin_inv_x(x: np.ndarray) -> float:
    """
    f(x) = sin(1/x) with f(0) = 0
    This function has infinitely many local minima as x approaches 0
    """
    if np.abs(x[0]) < 1e-10:
        return 0.0
    return np.sin(1.0 / x[0])

def sin_inv_x_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of sin(1/x)
    df/dx = -cos(1/x) / x^2
    """
    if np.abs(x[0]) < 1e-10:
        return np.array([0.0])
    return np.array([-np.cos(1.0 / x[0]) / (x[0]**2)])

# ============================================================================
# OPTIMIZER IMPLEMENTATIONS (FROM SCRATCH)
# ============================================================================

class GradientDescent:
    """Standard Gradient Descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.name = "Gradient Descent"
    
    def optimize(self, x0: np.ndarray, grad_fn: Callable, func: Callable,
                 max_iters: int = 10000, tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        x = x0.copy().astype(float)
        history = [func(x)]
        x_history = [x.copy()]
        
        for i in range(max_iters):
            grad = grad_fn(x)
            x = x - self.lr * grad
            
            current_val = func(x)
            history.append(current_val)
            x_history.append(x.copy())
            
            # Convergence check
            if len(history) > 1 and np.abs(history[-1] - history[-2]) < tol:
                break
            
            # Divergence check
            if np.abs(current_val) > 1e10 or np.isnan(current_val):
                break
        
        return x, history, x_history


class SGDMomentum:
    """Stochastic Gradient Descent with Momentum"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.name = "SGD with Momentum"
    
    def optimize(self, x0: np.ndarray, grad_fn: Callable, func: Callable,
                 max_iters: int = 10000, tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        x = x0.copy().astype(float)
        velocity = np.zeros_like(x)
        history = [func(x)]
        x_history = [x.copy()]
        
        for i in range(max_iters):
            grad = grad_fn(x)
            # Add small noise for stochastic behavior
            noise = np.random.normal(0, 0.01, size=grad.shape)
            grad_noisy = grad + noise
            
            velocity = self.momentum * velocity - self.lr * grad_noisy
            x = x + velocity
            
            current_val = func(x)
            history.append(current_val)
            x_history.append(x.copy())
            
            if len(history) > 1 and np.abs(history[-1] - history[-2]) < tol:
                break
            
            if np.abs(current_val) > 1e10 or np.isnan(current_val):
                break
        
        return x, history, x_history


class Adam:
    """Adam optimizer - Adaptive Moment Estimation"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = "Adam"
    
    def optimize(self, x0: np.ndarray, grad_fn: Callable, func: Callable,
                 max_iters: int = 10000, tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        x = x0.copy().astype(float)
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        history = [func(x)]
        x_history = [x.copy()]
        
        for t in range(1, max_iters + 1):
            grad = grad_fn(x)
            
            # Update biased first moment estimate
            m = self.beta1 * m + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            # Update parameters
            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
            current_val = func(x)
            history.append(current_val)
            x_history.append(x.copy())
            
            if len(history) > 1 and np.abs(history[-1] - history[-2]) < tol:
                break
            
            if np.abs(current_val) > 1e10 or np.isnan(current_val):
                break
        
        return x, history, x_history


class RMSprop:
    """RMSprop optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.99, 
                 epsilon: float = 1e-8):
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.name = "RMSprop"
    
    def optimize(self, x0: np.ndarray, grad_fn: Callable, func: Callable,
                 max_iters: int = 10000, tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        x = x0.copy().astype(float)
        cache = np.zeros_like(x)
        history = [func(x)]
        x_history = [x.copy()]
        
        for i in range(max_iters):
            grad = grad_fn(x)
            
            # Update cache (exponentially decaying average of squared gradients)
            cache = self.decay_rate * cache + (1 - self.decay_rate) * (grad ** 2)
            
            # Update parameters
            x = x - self.lr * grad / (np.sqrt(cache) + self.epsilon)
            
            current_val = func(x)
            history.append(current_val)
            x_history.append(x.copy())
            
            if len(history) > 1 and np.abs(history[-1] - history[-2]) < tol:
                break
            
            if np.abs(current_val) > 1e10 or np.isnan(current_val):
                break
        
        return x, history, x_history


class Adagrad:
    """Adagrad optimizer - Adaptive Gradient Algorithm"""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.name = "Adagrad"
    
    def optimize(self, x0: np.ndarray, grad_fn: Callable, func: Callable,
                 max_iters: int = 10000, tol: float = 1e-8) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        x = x0.copy().astype(float)
        cache = np.zeros_like(x)  # Sum of squared gradients
        history = [func(x)]
        x_history = [x.copy()]
        
        for i in range(max_iters):
            grad = grad_fn(x)
            
            # Accumulate squared gradients
            cache = cache + grad ** 2
            
            # Update parameters
            x = x - self.lr * grad / (np.sqrt(cache) + self.epsilon)
            
            current_val = func(x)
            history.append(current_val)
            x_history.append(x.copy())
            
            if len(history) > 1 and np.abs(history[-1] - history[-2]) < tol:
                break
            
            if np.abs(current_val) > 1e10 or np.isnan(current_val):
                break
        
        return x, history, x_history


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiments():
    """Run all optimization experiments and generate plots"""
    
    learning_rates = [0.01, 0.05, 0.1]
    max_iters = 10000
    
    # Results storage
    results = {
        'rosenbrock': {},
        'sin_inv_x': {}
    }
    
    # ========================================================================
    # EXPERIMENT 1: ROSENBROCK FUNCTION
    # ========================================================================
    print("=" * 70)
    print("EXPERIMENT 1: ROSENBROCK FUNCTION")
    print("f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print("Global minimum at (1, 1) with f(1,1) = 0")
    print("=" * 70)
    
    # Initial point
    x0_rosenbrock = np.array([-1.5, 1.5])
    
    for lr in learning_rates:
        print(f"\n--- Learning Rate: {lr} ---")
        results['rosenbrock'][lr] = {}
        
        optimizers = [
            GradientDescent(learning_rate=lr),
            SGDMomentum(learning_rate=lr),
            Adam(learning_rate=lr),
            RMSprop(learning_rate=lr),
            Adagrad(learning_rate=lr)
        ]
        
        for opt in optimizers:
            start_time = time.time()
            x_final, history, x_history = opt.optimize(
                x0_rosenbrock, rosenbrock_gradient, rosenbrock, max_iters
            )
            elapsed_time = time.time() - start_time
            
            results['rosenbrock'][lr][opt.name] = {
                'x_final': x_final,
                'f_final': history[-1],
                'history': history,
                'x_history': x_history,
                'iterations': len(history) - 1,
                'time': elapsed_time
            }
            
            print(f"{opt.name:25s}: x* = [{x_final[0]:.6f}, {x_final[1]:.6f}], "
                  f"f(x*) = {history[-1]:.6e}, Iterations = {len(history)-1}, "
                  f"Time = {elapsed_time:.4f}s")
    
    # ========================================================================
    # EXPERIMENT 2: SIN(1/X) FUNCTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SIN(1/X) FUNCTION")
    print("f(x) = sin(1/x) with f(0) = 0")
    print("Note: This function has infinitely many local minima")
    print("=" * 70)
    
    # Initial point (away from 0)
    x0_sin = np.array([0.5])
    
    for lr in learning_rates:
        print(f"\n--- Learning Rate: {lr} ---")
        results['sin_inv_x'][lr] = {}
        
        # Use smaller learning rates for this function
        adjusted_lr = lr * 0.1
        
        optimizers = [
            GradientDescent(learning_rate=adjusted_lr),
            SGDMomentum(learning_rate=adjusted_lr),
            Adam(learning_rate=adjusted_lr),
            RMSprop(learning_rate=adjusted_lr),
            Adagrad(learning_rate=adjusted_lr)
        ]
        
        for opt in optimizers:
            start_time = time.time()
            x_final, history, x_history = opt.optimize(
                x0_sin, sin_inv_x_gradient, sin_inv_x, max_iters=5000
            )
            elapsed_time = time.time() - start_time
            
            results['sin_inv_x'][lr][opt.name] = {
                'x_final': x_final,
                'f_final': history[-1],
                'history': history,
                'x_history': x_history,
                'iterations': len(history) - 1,
                'time': elapsed_time
            }
            
            print(f"{opt.name:25s}: x* = {x_final[0]:.6f}, "
                  f"f(x*) = {history[-1]:.6e}, Iterations = {len(history)-1}, "
                  f"Time = {elapsed_time:.4f}s")
    
    return results


def create_plots(results: Dict):
    """Generate all required plots"""
    
    learning_rates = [0.01, 0.05, 0.1]
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    optimizer_names = ['Gradient Descent', 'SGD with Momentum', 'Adam', 'RMSprop', 'Adagrad']
    
    # ========================================================================
    # PLOT 1: CONVERGENCE CURVES FOR ROSENBROCK
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Rosenbrock Function: Convergence Behavior', fontsize=14, fontweight='bold')
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        for i, opt_name in enumerate(optimizer_names):
            history = results['rosenbrock'][lr][opt_name]['history']
            # Clip for better visualization
            history_clipped = np.clip(history, 1e-10, 1e6)
            ax.semilogy(history_clipped, label=opt_name, color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Iterations', fontsize=11)
        ax.set_ylabel('f(x) [log scale]', fontsize=11)
        ax.set_title(f'Learning Rate = {lr}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(5000, max(len(results['rosenbrock'][lr][opt]['history']) for opt in optimizer_names))])
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/rosenbrock_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 2: CONVERGENCE CURVES FOR SIN(1/X)
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Sin(1/x) Function: Convergence Behavior', fontsize=14, fontweight='bold')
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        for i, opt_name in enumerate(optimizer_names):
            history = results['sin_inv_x'][lr][opt_name]['history']
            ax.plot(history, label=opt_name, color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Iterations', fontsize=11)
        ax.set_ylabel('f(x)', fontsize=11)
        ax.set_title(f'Learning Rate = {lr}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/sin_inv_x_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 3: TIME COMPARISON
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rosenbrock time comparison
    ax = axes[0]
    x_pos = np.arange(len(optimizer_names))
    width = 0.25
    
    for idx, lr in enumerate(learning_rates):
        times = [results['rosenbrock'][lr][opt]['time'] for opt in optimizer_names]
        ax.bar(x_pos + idx * width, times, width, label=f'lr={lr}')
    
    ax.set_xlabel('Optimizer', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Rosenbrock: Time Taken by Each Optimizer', fontsize=12)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(['GD', 'SGD-M', 'Adam', 'RMSprop', 'Adagrad'], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Sin(1/x) time comparison
    ax = axes[1]
    for idx, lr in enumerate(learning_rates):
        times = [results['sin_inv_x'][lr][opt]['time'] for opt in optimizer_names]
        ax.bar(x_pos + idx * width, times, width, label=f'lr={lr}')
    
    ax.set_xlabel('Optimizer', fontsize=11)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Sin(1/x): Time Taken by Each Optimizer', fontsize=12)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(['GD', 'SGD-M', 'Adam', 'RMSprop', 'Adagrad'], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 4: TRAJECTORY ON ROSENBROCK CONTOUR
    # ========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Rosenbrock Function: Optimization Trajectories', fontsize=14, fontweight='bold')
    
    # Create contour
    x_range = np.linspace(-2, 2, 200)
    y_range = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        
        # Plot contour
        levels = np.logspace(-1, 3, 20)
        ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
        
        # Plot trajectories
        for i, opt_name in enumerate(optimizer_names):
            x_hist = np.array(results['rosenbrock'][lr][opt_name]['x_history'])
            if len(x_hist) > 1:
                # Limit points for clarity
                step = max(1, len(x_hist) // 100)
                ax.plot(x_hist[::step, 0], x_hist[::step, 1], 'o-', 
                       color=colors[i], label=opt_name, markersize=2, linewidth=1, alpha=0.7)
        
        # Mark global minimum
        ax.plot(1, 1, 'r*', markersize=15, label='Global min (1,1)')
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Learning Rate = {lr}', fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 3])
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/rosenbrock_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 5: HYPERPARAMETER IMPACT ANALYSIS
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Impact of Learning Rate on Final Function Value', fontsize=14, fontweight='bold')
    
    for i, opt_name in enumerate(optimizer_names[:3]):
        # Rosenbrock
        ax = axes[0, i]
        final_vals = [results['rosenbrock'][lr][opt_name]['f_final'] for lr in learning_rates]
        ax.bar(range(len(learning_rates)), np.clip(final_vals, 1e-10, 1e4), color=colors[i])
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([str(lr) for lr in learning_rates])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Final f(x) [clipped]')
        ax.set_title(f'Rosenbrock - {opt_name}')
        ax.set_yscale('log')
    
    for i, opt_name in enumerate(optimizer_names[3:]):
        ax = axes[1, i]
        final_vals = [results['rosenbrock'][lr][opt_name]['f_final'] for lr in learning_rates]
        ax.bar(range(len(learning_rates)), np.clip(final_vals, 1e-10, 1e4), color=colors[i+3])
        ax.set_xticks(range(len(learning_rates)))
        ax.set_xticklabels([str(lr) for lr in learning_rates])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Final f(x) [clipped]')
        ax.set_title(f'Rosenbrock - {opt_name}')
        ax.set_yscale('log')
    
    # Fill remaining subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/GroupK_Assignment2/hyperparameter_impact.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nAll plots saved successfully!")


def create_summary_table(results: Dict) -> str:
    """Create a formatted summary table of results"""
    
    learning_rates = [0.01, 0.05, 0.1]
    optimizer_names = ['Gradient Descent', 'SGD with Momentum', 'Adam', 'RMSprop', 'Adagrad']
    
    summary = []
    summary.append("\n" + "=" * 100)
    summary.append("SUMMARY TABLE: ROSENBROCK FUNCTION (Fixed Learning Rate = 0.01)")
    summary.append("=" * 100)
    summary.append(f"{'Optimizer':<25} {'Final x*':<30} {'f(x*)':<15} {'Iterations':<12} {'Time (s)':<10}")
    summary.append("-" * 100)
    
    lr = 0.01
    for opt_name in optimizer_names:
        res = results['rosenbrock'][lr][opt_name]
        x_str = f"[{res['x_final'][0]:.6f}, {res['x_final'][1]:.6f}]"
        summary.append(f"{opt_name:<25} {x_str:<30} {res['f_final']:<15.6e} {res['iterations']:<12} {res['time']:<10.4f}")
    
    summary.append("\n" + "=" * 100)
    summary.append("SUMMARY TABLE: SIN(1/X) FUNCTION (Fixed Learning Rate = 0.01)")
    summary.append("=" * 100)
    summary.append(f"{'Optimizer':<25} {'Final x*':<15} {'f(x*)':<15} {'Iterations':<12} {'Time (s)':<10}")
    summary.append("-" * 100)
    
    for opt_name in optimizer_names:
        res = results['sin_inv_x'][lr][opt_name]
        summary.append(f"{opt_name:<25} {res['x_final'][0]:<15.6f} {res['f_final']:<15.6e} {res['iterations']:<12} {res['time']:<10.4f}")
    
    return "\n".join(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Task 1: Optimizer Performance on Non-Convex Functions")
    print("=" * 70)
    print("Group K - AI Assignment 2")
    print("=" * 70)
    
    # Run experiments
    results = run_experiments()
    
    # Create plots
    create_plots(results)
    
    # Print summary table
    summary = create_summary_table(results)
    print(summary)
    
    # Save summary to file
    with open('/home/claude/GroupK_Assignment2/task1_results.txt', 'w') as f:
        f.write("Task 1: Optimizer Performance on Non-Convex Functions\n")
        f.write("Group K - AI Assignment 2\n")
        f.write("=" * 70 + "\n\n")
        f.write(summary)
    
    print("\n" + "=" * 70)
    print("Task 1 completed successfully!")
    print("Plots saved to /home/claude/GroupK_Assignment2/")
    print("=" * 70)
