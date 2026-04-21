"""
task1_policy_evaluation.py — Iterative Policy Evaluation
=========================================================
Course: MA3206 Artificial Intelligence | Assignment 5
Group: K

Tasks covered:
  1.1  Build and verify the MDP
  1.2  Implement policy_evaluation()
  1.3  Bar chart of V^pi(s)
"""

import numpy as np
import matplotlib.pyplot as plt
from mdp_setup import STATES, ACTIONS, P, R, GAMMA, THETA, NUM_STATES, verify_mdp


# =====================================================================
# Task 1.1 — Verify MDP setup
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 1.1 — MDP Setup and Verification")
print("=" * 65)
verify_mdp()


# =====================================================================
# Task 1.2 — Implement policy_evaluation()
# =====================================================================

def policy_evaluation(P, R, policy, gamma, theta):
    """
    Iterative Policy Evaluation using the Bellman Expectation Equation.

    Given a fixed deterministic policy pi, this function computes V^pi(s)
    for every state by repeatedly applying:

        V(s) <- sum_{s'} P(s' | s, pi(s)) * [ R(s, pi(s)) + gamma * V(s') ]

    Parameters
    ----------
    P      : ndarray, shape (|S|, |A|, |S|) — transition probabilities
    R      : ndarray, shape (|S|, |A|)       — expected immediate rewards
    policy : ndarray, shape (|S|,)           — action index for each state
    gamma  : float                           — discount factor
    theta  : float                           — convergence threshold

    Returns
    -------
    V : ndarray, shape (|S|,) — converged state-value function V^pi
    """
    n_states = P.shape[0]
    V = np.zeros(n_states)   # Initialise V(s) = 0 for all states
    iterations = 0

    while True:
        V_new = np.zeros(n_states)

        for s in range(n_states):
            a = policy[s]
            # Bellman expectation update
            for s_prime in range(n_states):
                V_new[s] += P[s, a, s_prime] * (R[s, a] + gamma * V[s_prime])

        # Check convergence: max absolute change across all states
        delta = np.max(np.abs(V_new - V))
        V = V_new.copy()
        iterations += 1

        if delta < theta:
            break

    print(f"  Policy Evaluation converged in {iterations} iterations (theta={theta}).")
    return V


# =====================================================================
# Evaluate the specified policy:  High->Search, Low->Wait, Charging->Wait
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 1.2 — Policy Evaluation Results")
print("=" * 65)

policy_1 = np.array([0, 1, 1])   # High->Search(0), Low->Wait(1), Charging->Wait(1)
print(f"\n  Policy being evaluated:")
for s in range(NUM_STATES):
    print(f"    pi({STATES[s]:>8s}) = {ACTIONS[policy_1[s]]}")

print()
V_pi = policy_evaluation(P, R, policy_1, GAMMA, THETA)

print(f"\n  Converged V^pi(s):")
print(f"    {'State':>10s}   {'V^pi(s)':>12s}")
print(f"    {'-'*10}   {'-'*12}")
for s in range(NUM_STATES):
    print(f"    {STATES[s]:>10s}   {V_pi[s]:>12.6f}")

# Manual verification for Low state:
# pi(Low) = Wait, so V(Low) = R(Low,Wait) + gamma*V(Low)
# => V(Low) = 1.0 + 0.9*V(Low) => 0.1*V(Low) = 1.0 => V(Low) = 10.0  ✓
print(f"\n  Manual check: V(Low) should be 1/(1-0.9) = 10.0 -> Got {V_pi[1]:.4f} ✓")

# Manual verification for Charging state:
# pi(Charging) = Wait, transitions to High with R=0
# V(Charging) = 0 + 0.9*V(High) = 0.9 * 18.1081 = 16.2973  ✓
print(f"  Manual check: V(Charging) = 0.9 * V(High) = 0.9 * {V_pi[0]:.4f} = {0.9*V_pi[0]:.4f}")
print(f"                Got V(Charging) = {V_pi[2]:.4f} ✓")


# =====================================================================
# Task 1.3 — Bar chart of V^pi(s)
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 1.3 — Generating Bar Chart")
print("=" * 65)

fig, ax = plt.subplots(figsize=(8, 5.5))
colors_bar = ['#2196F3', '#FF9800', '#4CAF50']
bars = ax.bar(STATES, V_pi, color=colors_bar, edgecolor='black', linewidth=0.8, width=0.5)

# Annotate bars with values
for bar, val in zip(bars, V_pi):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('State', fontsize=13)
ax.set_ylabel(r'$V^{\pi}(s)$', fontsize=14)
ax.set_title('Task 1.3: State Values under Policy π\n'
             r'$\pi$: High$\rightarrow$Search, Low$\rightarrow$Wait, Charging$\rightarrow$Wait',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, max(V_pi) * 1.18)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('task1_bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Bar chart saved as task1_bar_chart.png")

print("\n  Task 1 complete.\n")
