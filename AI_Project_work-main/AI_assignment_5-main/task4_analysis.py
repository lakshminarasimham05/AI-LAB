"""
task4_analysis.py — Analysis and Interpretation
==================================================
Course: MA3206 Artificial Intelligence | Assignment 5
Group: K

Tasks covered:
  4.1  Compare Value Iteration and Policy Iteration
  4.2  Convergence behavior analysis
  4.3  Optimal policy interpretation
  4.4  Practical insight

This file also generates a comparison plot of VI vs PI convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from mdp_setup import STATES, ACTIONS, P, R, GAMMA, THETA, NUM_STATES, NUM_ACTIONS

# Import functions from earlier tasks
from task1_policy_evaluation import policy_evaluation
from task2_value_iteration import value_iteration, extract_policy
from task3_policy_iteration import policy_iteration


# =====================================================================
# Re-run both algorithms to collect data
# =====================================================================
print("\n" + "=" * 65)
print("  TASK 4 — Analysis and Interpretation")
print("=" * 65)

print("\n  Running Value Iteration...")
V_star_vi, vi_history = value_iteration(P, R, GAMMA, THETA)
pi_vi = extract_policy(V_star_vi, P, R, GAMMA)

print("\n  Running Policy Iteration...")
pi_pi, V_star_pi, pi_policy_hist, pi_V_hist = policy_iteration(P, R, GAMMA, THETA)

# =====================================================================
# 4.1 — Compare VI and PI
# =====================================================================
print("\n" + "=" * 65)
print("  4.1  Comparison: Value Iteration vs Policy Iteration")
print("=" * 65)

vi_iters = len(vi_history) - 1
pi_iters = len(pi_V_hist)

print(f"""
  Value Iteration:
    - Converged in {vi_iters} Bellman optimality sweeps
    - Each sweep is one pass through all states, taking max over actions

  Policy Iteration:
    - Converged in {pi_iters} outer iteration(s)
    - Each outer iteration involves a full policy evaluation
      (which itself took ~133-142 inner sweeps) + one improvement step

  Which converges faster?
    In terms of outer iterations, Policy Iteration is dramatically faster
    ({pi_iters} vs {vi_iters}). This is because each PI step performs a
    *complete* policy evaluation before improving, so each improvement
    makes a large, well-informed jump in policy space.

    Value Iteration, on the other hand, makes small incremental updates
    at each sweep — it simultaneously tries to evaluate and improve,
    which is efficient per-sweep but requires many more sweeps.

    In terms of total Bellman update sweeps (inner + outer), both methods
    are comparable for this small MDP. For larger MDPs, PI typically needs
    fewer total sweeps because exact evaluation helps it converge faster.

  Both algorithms produce identical results:
    V*(High)     = {V_star_vi[0]:.6f}  (VI)  vs  {V_star_pi[0]:.6f}  (PI)
    V*(Low)      = {V_star_vi[1]:.6f}  (VI)  vs  {V_star_pi[1]:.6f}  (PI)
    V*(Charging) = {V_star_vi[2]:.6f}  (VI)  vs  {V_star_pi[2]:.6f}  (PI)
    pi*(High)     = {ACTIONS[pi_vi[0]]}  (VI)  vs  {ACTIONS[pi_pi[0]]}  (PI)
    pi*(Low)      = {ACTIONS[pi_vi[1]]}  (VI)  vs  {ACTIONS[pi_pi[1]]}  (PI)
    pi*(Charging) = {ACTIONS[pi_vi[2]]}  (VI)  vs  {ACTIONS[pi_pi[2]]}  (PI)
""")


# =====================================================================
# 4.2 — Convergence Behavior
# =====================================================================
print("=" * 65)
print("  4.2  Convergence Behavior")
print("=" * 65)

print(f"""
  Value Iteration Convergence (see Figure 2):
    - All three state values start at 0 and rise monotonically.
    - V(High) rises fastest because Search yields the highest immediate
      reward (+4) and has a 70% self-loop, creating compounding value.
    - V(Charging) initially lags behind V(Low) in early iterations
      because it earns 0 reward. However, once V(High) grows large,
      V(Charging) = gamma * V(High) overtakes V(Low).
    - By iteration ~40, values are within 1% of their final values.
      The remaining ~100 iterations refine the last decimal places.

  Why VI update differs from PE update:
    - Policy Evaluation uses:  V(s) <- E[R + gamma*V(s') | pi(s)]
      This is an *expectation* under a fixed policy — it can only
      compute V^pi, not V*.
    - Value Iteration uses:    V(s) <- max_a E[R + gamma*V(s') | a]
      The *max* operator explores all actions at each sweep, allowing
      VI to find V* directly without ever fixing a policy.
    - The max operator makes VI updates non-linear and typically causes
      slower convergence than the linear system solved by PE, but it
      reaches the globally optimal solution.
""")


# =====================================================================
# 4.3 — Optimal Policy Interpretation
# =====================================================================
print("=" * 65)
print("  4.3  Optimal Policy Interpretation")
print("=" * 65)

# Compute Q-values for detailed explanation
for s in range(NUM_STATES):
    q_search = sum(P[s, 0, sp] * (R[s, 0] + GAMMA * V_star_vi[sp]) for sp in range(NUM_STATES))
    q_wait   = sum(P[s, 1, sp] * (R[s, 1] + GAMMA * V_star_vi[sp]) for sp in range(NUM_STATES))
    print(f"\n  State: {STATES[s]}")
    print(f"    Q(Search) = {q_search:.4f}")
    print(f"    Q(Wait)   = {q_wait:.4f}")
    print(f"    Optimal:    {ACTIONS[pi_vi[s]]}  (margin: {abs(q_search - q_wait):.4f})")

print(f"""
  Interpretation:

  pi*(High) = Search:
    Q(Search) >> Q(Wait). The robot earns +4 per step with a 70% chance
    of staying High. Waiting earns only +1 with certainty. The expected
    future value of searching is much higher because the robot is likely
    to stay in the high-reward High state.

  pi*(Low) = Search:
    This is the non-obvious result. Even though there is a 40% chance
    of a -3 penalty (recharge), the expected immediate reward of Search
    is 0.4*(-3) + 0.6*(+4) = +1.2, which already exceeds Wait's +1.0.
    More importantly, the future values reached by Search are high:
    - 40% chance of going to High (V*=29.64) 
    - 60% chance of staying Low (V*=25.81)
    Weighted future: 0.4*29.64 + 0.6*25.81 = 27.34
    vs Wait's future: 1.0*25.81 = 25.81
    The combined immediate + discounted future strongly favors Search.

  pi*(Charging) = Wait:
    The only valid action. Search is not allowed in the Charging state.
    Wait transitions to High with 0 immediate reward, but the robot
    then has access to the lucrative Search action starting next step.
    V*(Charging) = 0 + 0.9 * V*(High) = 0.9 * 29.64 = 26.68.
""")


# =====================================================================
# 4.4 — Practical Insight
# =====================================================================
print("=" * 65)
print("  4.4  Practical Insight")
print("=" * 65)

print(f"""
  In real-world battery-powered robotic systems (warehouse AGVs, delivery
  drones, Mars rovers), battery management is a critical operational
  concern. The MDP framework reveals an important counter-intuitive
  insight: a naive "conserve energy whenever possible" strategy is
  suboptimal.

  The optimal policy says: keep working (Search) even when the battery
  is low, because the productivity gains (+4 reward) outweigh the
  occasional recharge penalty (-3). In practical terms, this means a
  delivery robot should continue its route even at low battery, because
  the expected revenue from completing deliveries exceeds the expected
  cost of an emergency recharge.

  This MDP approach is useful in practice for several reasons:
  1. The policy can be pre-computed offline and stored as a simple 
     lookup table (3 states -> 2 actions), requiring negligible runtime
     computation on embedded hardware.
  2. The discount factor gamma allows tuning the time horizon: a higher
     gamma makes the robot more forward-looking, while a lower gamma
     makes it prioritise immediate rewards.
  3. The framework naturally handles uncertainty through probability
     distributions, unlike deterministic rule-based systems.
  4. The approach scales to more complex settings: additional battery
     levels, multiple task types, or time-varying rewards can all be
     modelled by expanding the state/action spaces.
""")


# =====================================================================
# Bonus: Comparison visualization
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
colors_line = ['#1565C0', '#E65100', '#2E7D32']
markers = ['o', 's', '^']

# Left: VI convergence
ax = axes[0]
vi_arr = np.array(vi_history)
for s in range(NUM_STATES):
    ax.plot(range(len(vi_history)), vi_arr[:, s],
            label=STATES[s], color=colors_line[s], linewidth=1.8,
            marker=markers[s], markersize=3, markevery=5)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('V(s)', fontsize=12)
ax.set_title(f'Value Iteration ({vi_iters} sweeps)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Right: PI convergence
ax = axes[1]
pi_arr = np.array(pi_V_hist)
for s in range(NUM_STATES):
    ax.plot(range(1, len(pi_V_hist) + 1), pi_arr[:, s],
            label=STATES[s], color=colors_line[s], linewidth=2.5,
            marker=markers[s], markersize=8)
    for i, v in enumerate(pi_arr[:, s]):
        ax.annotate(f'{v:.2f}', xy=(i + 1, v),
                    xytext=(8, 5 if s == 0 else -12), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors_line[s])
ax.set_xlabel('Outer Iteration', fontsize=12)
ax.set_ylabel('V(s)', fontsize=12)
ax.set_title(f'Policy Iteration ({pi_iters} outer steps)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xticks(range(1, len(pi_V_hist) + 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('Task 4.1: Convergence Comparison — Value Iteration vs Policy Iteration',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('task4_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Comparison plot saved as task4_comparison.png")

print("\n" + "=" * 65)
print("  ALL TASKS COMPLETED SUCCESSFULLY")
print("=" * 65)
print()
