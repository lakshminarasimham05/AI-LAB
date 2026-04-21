"""
task3_policy_iteration.py — Policy Iteration
===============================================
Course: MA3206 Artificial Intelligence | Assignment 5
Group: K

Tasks covered:
  3.1  Implement policy_improvement()
  3.2  Implement full policy_iteration() loop
  3.3  Line plot of V(s) + heatmap of policy evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mdp_setup import STATES, ACTIONS, P, R, GAMMA, THETA, NUM_STATES, NUM_ACTIONS

# Import policy_evaluation from Task 1
from task1_policy_evaluation import policy_evaluation


# =====================================================================
# Task 3.1 — Policy Improvement
# =====================================================================

def policy_improvement(V, P, R, gamma, old_policy):
    """
    Improve the policy greedily based on the current value function V.

    For each state s, compute:
        Q(s, a) = sum_{s'} P(s'|s,a) * [ R(s,a) + gamma * V(s') ]
    then pick the action with the highest Q-value.

    NOTE: Action 'Search' is NOT allowed in the Charging state
          (as stated in the assignment). The Charging state is forced
          to always take 'Wait'.

    Parameters
    ----------
    V          : ndarray — current value function
    P, R       : MDP arrays
    gamma      : float
    old_policy : ndarray — previous policy (for stability check)

    Returns
    -------
    new_policy : ndarray — improved policy
    stable     : bool — True if policy did not change
    """
    n_states  = P.shape[0]
    n_actions = P.shape[1]
    new_policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q_values = np.zeros(n_actions)
        for a in range(n_actions):
            for s_prime in range(n_states):
                q_values[a] += P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])

        # Charging state: Search is NOT allowed
        if s == 2:  # Charging
            new_policy[s] = 1   # Must Wait
        else:
            new_policy[s] = np.argmax(q_values)

    stable = np.array_equal(new_policy, old_policy)
    return new_policy, stable


# =====================================================================
# Task 3.2 — Full Policy Iteration Loop
# =====================================================================

def policy_iteration(P, R, gamma, theta):
    """
    Full Policy Iteration algorithm.

    Steps:
      1. Start from initial policy: all states take Wait
      2. Policy Evaluation: compute V^pi for the current policy
      3. Policy Improvement: update policy greedily
      4. Repeat until stable (policy does not change)

    Returns
    -------
    policy         : final optimal policy
    V              : final value function V*
    policy_history : list of policies at each iteration
    V_history      : list of V arrays at each iteration
    """
    n_states = P.shape[0]

    # Initial policy: all states take Wait (action index 1)
    policy = np.ones(n_states, dtype=int)   # [Wait, Wait, Wait]

    policy_history = [policy.copy()]
    V_history = []
    iteration = 0

    while True:
        iteration += 1
        print(f"\n  {'─'*55}")
        print(f"  Policy Iteration — Step {iteration}")
        print(f"  {'─'*55}")

        # Display current policy
        print(f"  Current policy:")
        for s in range(n_states):
            print(f"    pi({STATES[s]:>8s}) = {ACTIONS[policy[s]]}")

        # Step A: Policy Evaluation
        print(f"\n  [Evaluation phase]")
        V = policy_evaluation(P, R, policy, gamma, theta)
        V_history.append(V.copy())

        print(f"\n  V^pi(s) after evaluation:")
        for s in range(n_states):
            print(f"    V({STATES[s]:>8s}) = {V[s]:.6f}")

        # Print Q-values to show why improvement happens
        print(f"\n  Q-values under current V:")
        print(f"    {'State':>10s}   {'Q(Search)':>10s}   {'Q(Wait)':>10s}   {'Greedy':>8s}")
        for s in range(n_states):
            q_s = sum(P[s, 0, sp] * (R[s, 0] + gamma * V[sp]) for sp in range(n_states))
            q_w = sum(P[s, 1, sp] * (R[s, 1] + gamma * V[sp]) for sp in range(n_states))
            best = "Search" if q_s > q_w and s != 2 else "Wait"
            marker = " *" if best != ACTIONS[policy[s]] else ""
            print(f"    {STATES[s]:>10s}   {q_s:>10.4f}   {q_w:>10.4f}   {best:>8s}{marker}")

        # Step B: Policy Improvement
        print(f"\n  [Improvement phase]")
        new_policy, stable = policy_improvement(V, P, R, gamma, policy)

        if stable:
            print(f"  Policy is STABLE — no changes. Converged!")
            policy_history.append(new_policy.copy())
            break
        else:
            changes = []
            for s in range(n_states):
                if policy[s] != new_policy[s]:
                    changes.append(f"{STATES[s]}: {ACTIONS[policy[s]]} -> {ACTIONS[new_policy[s]]}")
            print(f"  Policy CHANGED: {', '.join(changes)}")

        policy = new_policy.copy()
        policy_history.append(policy.copy())

    print(f"\n  Policy Iteration converged in {iteration} outer step(s).")
    return policy, V, policy_history, V_history


# =====================================================================
# Run Policy Iteration
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 3.2 — Policy Iteration (Full Run)")
print("=" * 65)

final_policy, V_star_pi, pi_policy_history, pi_V_history = \
    policy_iteration(P, R, GAMMA, THETA)

print(f"\n  {'='*55}")
print(f"  FINAL RESULTS")
print(f"  {'='*55}")
print(f"\n  Optimal policy from Policy Iteration:")
for s in range(NUM_STATES):
    print(f"    pi*({STATES[s]:>8s}) = {ACTIONS[final_policy[s]]}")

print(f"\n  Optimal V*(s) from Policy Iteration:")
for s in range(NUM_STATES):
    print(f"    V*({STATES[s]:>8s}) = {V_star_pi[s]:.6f}")


# =====================================================================
# Task 3.3(a) — Line plot of V(s) across iterations
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 3.3 — Generating Plots")
print("=" * 65)

fig, ax = plt.subplots(figsize=(9, 5.5))
V_hist_arr = np.array(pi_V_history)
colors_line = ['#1565C0', '#E65100', '#2E7D32']
markers = ['o', 's', '^']

for s in range(NUM_STATES):
    ax.plot(range(1, len(pi_V_history) + 1), V_hist_arr[:, s],
            label=f'{STATES[s]}', color=colors_line[s],
            marker=markers[s], markersize=8, linewidth=2.5)

    # Annotate with exact values
    for i, v in enumerate(V_hist_arr[:, s]):
        offset = 0.8 if s == 0 else (-1.2 if s == 2 else -1.5)
        ax.annotate(f'{v:.2f}', xy=(i + 1, v),
                    xytext=(8, offset * 10), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors_line[s])

ax.set_xlabel('Policy Iteration Step', fontsize=13)
ax.set_ylabel(r'$V^{\pi}(s)$', fontsize=14)
ax.set_title('Task 3.3(a): Policy Iteration — V(s) across Iteration Steps',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='center left')
ax.grid(alpha=0.3, linestyle='--')
ax.set_xticks(range(1, len(pi_V_history) + 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('task3_line_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Line plot saved as task3_line_plot.png")


# =====================================================================
# Task 3.3(b) — Heatmap / table of policy at each iteration
# =====================================================================
n_iters = len(pi_policy_history)
policy_matrix = np.array(pi_policy_history).T   # shape (3, n_iters)

fig, ax = plt.subplots(figsize=(max(7, n_iters * 2), 4.5))

cmap = ListedColormap(['#1565C0', '#FF9800'])
im = ax.imshow(policy_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Annotate cells with action names
for i in range(NUM_STATES):
    for j in range(n_iters):
        action_name = ACTIONS[policy_matrix[i, j]]
        ax.text(j, i, action_name,
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white')

ax.set_xticks(range(n_iters))
labels = ['Iter 0\n(Initial)'] + [f'Iter {k}' for k in range(1, n_iters)]
if n_iters > 1:
    labels[-1] = f'Iter {n_iters-1}\n(Stable)'
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks(range(NUM_STATES))
ax.set_yticklabels(STATES, fontsize=12)
ax.set_xlabel('Iteration', fontsize=13)
ax.set_ylabel('State', fontsize=13)
ax.set_title('Task 3.3(b): Policy Heatmap across Policy Iteration Steps\n'
             '(Blue = Search, Orange = Wait)',
             fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('task3_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Heatmap saved as task3_heatmap.png")

print("\n  Task 3 complete.\n")
