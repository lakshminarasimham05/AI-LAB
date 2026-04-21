"""
task2_value_iteration.py — Value Iteration
============================================
Course: MA3206 Artificial Intelligence | Assignment 5
Group: K

Tasks covered:
  2.1  Implement value_iteration()
  2.2  Extract the optimal policy
  2.3  Plot convergence of V(s) across iterations
"""

import numpy as np
import matplotlib.pyplot as plt
from mdp_setup import STATES, ACTIONS, P, R, GAMMA, THETA, NUM_STATES, NUM_ACTIONS


# =====================================================================
# Task 2.1 — Implement value_iteration()
# =====================================================================

def value_iteration(P, R, gamma, theta):
    """
    Value Iteration using the Bellman Optimality Equation.

    At each sweep, for every state s we compute:
        Q(s, a) = sum_{s'} P(s'|s,a) * [ R(s,a) + gamma * V(s') ]    for all a
        V(s)    = max_a Q(s, a)

    The process repeats until max_s |V_new(s) - V_old(s)| < theta.

    Parameters
    ----------
    P     : ndarray, shape (|S|, |A|, |S|)
    R     : ndarray, shape (|S|, |A|)
    gamma : float
    theta : float

    Returns
    -------
    V_star  : ndarray — converged optimal value function
    history : list of ndarrays — V after each iteration (for plotting)
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]
    V = np.zeros(n_states)
    history = [V.copy()]    # Store initial V (all zeros)
    iterations = 0

    while True:
        V_new = np.zeros(n_states)

        for s in range(n_states):
            q_values = np.zeros(n_actions)
            for a in range(n_actions):
                for s_prime in range(n_states):
                    q_values[a] += P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])
            V_new[s] = np.max(q_values)

        delta = np.max(np.abs(V_new - V))
        V = V_new.copy()
        history.append(V.copy())
        iterations += 1

        if delta < theta:
            break

    print(f"  Value Iteration converged in {iterations} iterations (theta={theta}).")
    return V, history


# =====================================================================
# Task 2.2 — Extract Optimal Policy
# =====================================================================

def extract_policy(V_star, P, R, gamma):
    """
    Extract the greedy policy from the optimal value function V*.

    For each state s:
        Q(s, a)  = sum_{s'} P(s'|s,a) * [ R(s,a) + gamma * V*(s') ]
        pi*(s)   = argmax_a Q(s, a)

    Parameters
    ----------
    V_star : ndarray — optimal value function
    P, R   : MDP arrays
    gamma  : float

    Returns
    -------
    policy : ndarray of action indices
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]
    policy    = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for s_prime in range(n_states):
                q_values[a] += P[s, a, s_prime] * (R[s, a] + gamma * V_star[s_prime])
        policy[s] = np.argmax(q_values)

    return policy


# =====================================================================
# Run Value Iteration
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 2.1 — Value Iteration")
print("=" * 65)

V_star, vi_history = value_iteration(P, R, GAMMA, THETA)

print(f"\n  Converged V*(s):")
print(f"    {'State':>10s}   {'V*(s)':>12s}")
print(f"    {'-'*10}   {'-'*12}")
for s in range(NUM_STATES):
    print(f"    {STATES[s]:>10s}   {V_star[s]:>12.6f}")

# Print Q-values for transparency
print(f"\n  Q-values at convergence:")
print(f"    {'State':>10s}   {'Q(s,Search)':>12s}   {'Q(s,Wait)':>12s}   {'Best Action':>12s}")
print(f"    {'-'*10}   {'-'*12}   {'-'*12}   {'-'*12}")
for s in range(NUM_STATES):
    q_search = sum(P[s, 0, sp] * (R[s, 0] + GAMMA * V_star[sp]) for sp in range(NUM_STATES))
    q_wait   = sum(P[s, 1, sp] * (R[s, 1] + GAMMA * V_star[sp]) for sp in range(NUM_STATES))
    best = "Search" if q_search >= q_wait else "Wait"
    print(f"    {STATES[s]:>10s}   {q_search:>12.6f}   {q_wait:>12.6f}   {best:>12s}")


# =====================================================================
# Extract and display optimal policy
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 2.2 — Optimal Policy Extraction")
print("=" * 65)

optimal_policy = extract_policy(V_star, P, R, GAMMA)

print(f"\n  Optimal policy pi*(s):")
for s in range(NUM_STATES):
    print(f"    pi*({STATES[s]:>8s}) = {ACTIONS[optimal_policy[s]]}")


# =====================================================================
# Task 2.3 — Convergence plot
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 2.3 — Convergence Plot")
print("=" * 65)

fig, ax = plt.subplots(figsize=(10, 5.5))
history_arr = np.array(vi_history)
colors_line = ['#1565C0', '#E65100', '#2E7D32']
markers = ['o', 's', '^']

for s in range(NUM_STATES):
    ax.plot(range(len(vi_history)), history_arr[:, s],
            label=f'{STATES[s]}  (converges to {V_star[s]:.2f})',
            color=colors_line[s], marker=markers[s], markersize=3,
            linewidth=1.8, markevery=5)

ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('V(s)', fontsize=13)
ax.set_title('Task 2.3: Value Iteration — Convergence of V(s) over Iterations',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotations for final values
for s in range(NUM_STATES):
    ax.annotate(f'{V_star[s]:.2f}',
                xy=(len(vi_history)-1, V_star[s]),
                xytext=(10, 0), textcoords='offset points',
                fontsize=9, fontweight='bold', color=colors_line[s])

plt.tight_layout()
plt.savefig('task2_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Convergence plot saved as task2_convergence.png")

# Also print some iteration snapshots for the report
print(f"\n  Selected iteration snapshots:")
print(f"    {'Iter':>6s}   {'V(High)':>10s}   {'V(Low)':>10s}   {'V(Charging)':>12s}")
print(f"    {'-'*6}   {'-'*10}   {'-'*10}   {'-'*12}")
for i in [0, 1, 2, 5, 10, 20, 50, 100, len(vi_history)-1]:
    if i < len(vi_history):
        print(f"    {i:>6d}   {vi_history[i][0]:>10.4f}   {vi_history[i][1]:>10.4f}   {vi_history[i][2]:>12.4f}")

print("\n  Task 2 complete.\n")
