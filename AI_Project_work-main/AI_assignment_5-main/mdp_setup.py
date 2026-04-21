"""
mdp_setup.py — MDP Definition for the Battery Robot Problem
============================================================
Course: MA3206 Artificial Intelligence | Assignment 5
Group: K

This module defines the MDP components (states, actions, transition
probabilities, rewards) and is imported by all task-specific files.
"""

import numpy as np

# ---------------------------------------------------------------------------
# State and Action spaces
# ---------------------------------------------------------------------------
STATES  = ['High', 'Low', 'Charging']   # indices 0, 1, 2
ACTIONS = ['Search', 'Wait']             # indices 0, 1

NUM_STATES  = len(STATES)
NUM_ACTIONS = len(ACTIONS)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
GAMMA = 0.9          # Discount factor
THETA = 1e-6         # Convergence threshold

# ---------------------------------------------------------------------------
# Transition Probability Array  P[s, a, s']  shape (3, 2, 3)
# ---------------------------------------------------------------------------
P = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))

# From High (s=0)
P[0, 0, 0] = 0.7    # High --Search--> High   (prob 0.7)
P[0, 0, 1] = 0.3    # High --Search--> Low    (prob 0.3)
P[0, 1, 0] = 1.0    # High --Wait  --> High   (prob 1.0)

# From Low (s=1)
P[1, 0, 0] = 0.4    # Low  --Search--> High   (prob 0.4, recharge penalty)
P[1, 0, 1] = 0.6    # Low  --Search--> Low    (prob 0.6)
P[1, 1, 1] = 1.0    # Low  --Wait  --> Low    (prob 1.0)

# From Charging (s=2)
P[2, 0, 2] = 1.0    # Charging --Search--> Charging  (self-loop; action disallowed by policy)
P[2, 1, 0] = 1.0    # Charging --Wait  --> High      (prob 1.0)

# ---------------------------------------------------------------------------
# Reward Array  R[s, a]  shape (3, 2)
# Computed as:  R[s,a] = sum_{s'} P(s'|s,a) * R(s,a,s')
# ---------------------------------------------------------------------------
# Individual transition rewards from the table:
#   High,Search -> High:  +4      High,Search -> Low:   +4
#   High,Wait   -> High:  +1
#   Low,Search  -> High:  -3      Low,Search  -> Low:   +4
#   Low,Wait    -> Low:   +1
#   Charging,Wait -> High: 0
#
# Expected rewards:
#   R[High, Search]     = 0.7*(+4) + 0.3*(+4) = 4.0
#   R[High, Wait]       = 1.0*(+1)            = 1.0
#   R[Low, Search]      = 0.4*(-3) + 0.6*(+4) = -1.2 + 2.4 = 1.2
#   R[Low, Wait]        = 1.0*(+1)            = 1.0
#   R[Charging, Search] = 1.0*(0)             = 0.0   (action not allowed)
#   R[Charging, Wait]   = 1.0*(0)             = 0.0

R = np.zeros((NUM_STATES, NUM_ACTIONS))
R[0, 0] = 4.0    # High,   Search
R[0, 1] = 1.0    # High,   Wait
R[1, 0] = 1.2    # Low,    Search
R[1, 1] = 1.0    # Low,    Wait
R[2, 0] = 0.0    # Charging, Search  (not allowed)
R[2, 1] = 0.0    # Charging, Wait


# ---------------------------------------------------------------------------
# Verification utility
# ---------------------------------------------------------------------------
def verify_mdp():
    """Print verification checks for the MDP setup."""
    print("=" * 65)
    print("  MDP SETUP VERIFICATION")
    print("=" * 65)

    print("\n  Transition probability row sums (must all be 1.0):")
    all_ok = True
    for s in range(NUM_STATES):
        for a in range(NUM_ACTIONS):
            row_sum = P[s, a, :].sum()
            status = "OK" if np.isclose(row_sum, 1.0) else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"    P[{STATES[s]:>8s}, {ACTIONS[a]:>6s}, :] = {row_sum:.4f}  [{status}]")
    print(f"\n  All row sums valid: {all_ok}")

    print("\n  Reward array R[s, a]:")
    print(f"    {'':>10s}  {'Search':>8s}  {'Wait':>8s}")
    for s in range(NUM_STATES):
        print(f"    {STATES[s]:>10s}  {R[s,0]:>+8.1f}  {R[s,1]:>+8.1f}")

    print("\n  Full P array (transition probabilities):")
    for s in range(NUM_STATES):
        for a in range(NUM_ACTIONS):
            probs = ", ".join([f"{STATES[sp]}:{P[s,a,sp]:.1f}" for sp in range(NUM_STATES) if P[s,a,sp] > 0])
            print(f"    P[{STATES[s]:>8s}, {ACTIONS[a]:>6s}] -> {probs}")

    print("=" * 65)


if __name__ == "__main__":
    verify_mdp()
