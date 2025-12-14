# energy_game_selfplay.py

import numpy as np
from enum import IntEnum

# =========================
# 1. Game definition
# =========================

class Action(IntEnum):
    STORE = 0        # S
    ATTACK_SMALL = 1 # AS
    ATTACK_BIG = 2   # AB
    DEFEND_SMALL = 3 # DS
    DEFEND_BIG = 4   # DB


ACTION_NAMES = {
    Action.STORE: "STORE",
    Action.ATTACK_SMALL: "ATTACK_SMALL",
    Action.ATTACK_BIG: "ATTACK_BIG",
    Action.DEFEND_SMALL: "DEFEND_SMALL",
    Action.DEFEND_BIG: "DEFEND_BIG",
}


class EnergyGameEnv:
    """
    Two-player simultaneous-move game.
    Each player has energy in {0, 1, 2}.
    State is (energy_p0, energy_p1).
    Episode ends immediately when someone is hit.
    """

    def __init__(self, step_penalty=0.01):
        self.max_energy = 2
        self.state = (0, 0)
        self.done = False
        self.step_penalty = step_penalty

    def reset(self):
        self.state = (0, 0)
        self.done = False
        return self.state

    def _attack_size(self, energy, action):
        """
        Returns (attack_size, energy_cost):
          attack_size in {0, 1, 2} for none / small / big
        Assumes action is valid for given energy.
        """
        if action == Action.ATTACK_SMALL:
            return 1, 1
        if action == Action.ATTACK_BIG:
            return 2, 2
        return 0, 0

    def _defense_size(self, action):
        """
        Returns defense_size in {0, 1, 2} for none / small / big defense.
        """
        if action == Action.DEFEND_SMALL:
            return 1
        if action == Action.DEFEND_BIG:
            return 2
        return 0

    def step(self, action_p0, action_p1):
        """
        Takes joint action (action_p0, action_p1),
        returns: next_state, (reward_p0, reward_p1), done, info
        """
        if self.done:
            raise RuntimeError("Episode already finished. Call reset().")

        e0, e1 = self.state

        atk0, cost0 = self._attack_size(e0, action_p0)
        atk1, cost1 = self._attack_size(e1, action_p1)

        def0 = self._defense_size(action_p0)
        def1 = self._defense_size(action_p1)

        hit0 = False
        hit1 = False

        # --- Resolve combat ---

        # Case 1: both attack
        if atk0 > 0 and atk1 > 0:
            if atk0 == atk1:
                # same size -> cancel
                hit0 = False
                hit1 = False
            else:
                # bigger attack wins
                if atk0 > atk1:
                    hit1 = True
                else:
                    hit0 = True

        # Case 2: only p0 attacks
        elif atk0 > 0 and atk1 == 0:
            if def1 == atk0:
                hit1 = False  # blocked
            else:
                hit1 = True

        # Case 3: only p1 attacks
        elif atk1 > 0 and atk0 == 0:
            if def0 == atk1:
                hit0 = False
            else:
                hit0 = True

        # Case 4: nobody attacks -> nothing happens

        # --- Rewards & termination ---
        r0 = 0.0
        r1 = 0.0
        if hit0 and not hit1:
            r0 = -1.0
            r1 = +1.0
            self.done = True
        elif hit1 and not hit0:
            r0 = +1.0
            r1 = -1.0
            self.done = True
        elif hit0 and hit1:
            r0 = 0.0
            r1 = 0.0
            self.done = True

        # small time penalty if game continues
        if not self.done and self.step_penalty is not None and self.step_penalty > 0:
            r0 -= self.step_penalty
            r1 -= self.step_penalty

        # --- Update energies ---
        new_e0 = e0
        new_e1 = e1

        # Player 0
        if action_p0 == Action.STORE:
            new_e0 = min(self.max_energy, e0 + 1)
        elif atk0 > 0:
            new_e0 = e0 - cost0

        # Player 1
        if action_p1 == Action.STORE:
            new_e1 = min(self.max_energy, e1 + 1)
        elif atk1 > 0:
            new_e1 = e1 - cost1

        self.state = (new_e0, new_e1)
        return self.state, (r0, r1), self.done, {}


# =========================
# 2. Tabular Q-learning + masks
# =========================

NUM_ENERGY = 3          # energies: 0,1,2
NUM_ACTIONS = 5         # STORE, AS, AB, DS, DB
NUM_STATES = NUM_ENERGY * NUM_ENERGY  # (my_e, opp_e)


def state_to_idx(my_energy, opp_energy):
    """Map (my_energy, opp_energy) -> integer 0..8."""
    return my_energy * NUM_ENERGY + opp_energy


def valid_actions_for_state(my_energy, opp_energy):
    """
    Valid actions depend on BOTH my_energy and opp_energy.

    - STORE always valid.
    - ATTACK_SMALL only if my_energy >= 1
    - ATTACK_BIG   only if my_energy >= 2
    - DEFEND_SMALL only if opponent could small-attack (opp_energy >= 1)
    - DEFEND_BIG   only if opponent could big-attack   (opp_energy >= 2)
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)

    # Always allowed
    mask[Action.STORE] = True

    # Attacks depend on my energy
    if my_energy >= 1:
        mask[Action.ATTACK_SMALL] = True
    if my_energy >= 2:
        mask[Action.ATTACK_BIG] = True

    # Defenses depend on opponent energy (what they COULD do)
    if opp_energy >= 1:
        mask[Action.DEFEND_SMALL] = True
    if opp_energy >= 2:
        mask[Action.DEFEND_BIG] = True

    return mask


def epsilon_greedy(Q, s_idx, epsilon, valid_mask):
    """
    ε-greedy over *valid* actions only.
    valid_mask: boolean mask of shape (NUM_ACTIONS,)
    """
    valid_indices = np.where(valid_mask)[0]
    if np.random.rand() < epsilon:
        return int(np.random.choice(valid_indices))
    q_vals = Q[s_idx, valid_indices]
    best = valid_indices[np.argmax(q_vals)]
    return int(best)


def train_self_play(
    num_episodes=50000,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=0.2,
    epsilon_end=0.01,
    seed=0,
    step_penalty=0.01,
):
    """
    Train two Q-learning agents in self-play.
    Returns Q0, Q1 (Q-tables for each player).
    """
    np.random.seed(seed)
    env = EnergyGameEnv(step_penalty=step_penalty)

    Q0 = np.zeros((NUM_STATES, NUM_ACTIONS))  # player 0
    Q1 = np.zeros((NUM_STATES, NUM_ACTIONS))  # player 1

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        # Linear epsilon decay
        frac = episode / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        while not done:
            e0, e1 = state

            # Each player views the state from their perspective:
            s0 = state_to_idx(e0, e1)  # player 0: (my_energy=e0, opp_energy=e1)
            s1 = state_to_idx(e1, e0)  # player 1: (my_energy=e1, opp_energy=e0)

            # Valid actions for each player given (my_e, opp_e)
            valid0 = valid_actions_for_state(e0, e1)
            valid1 = valid_actions_for_state(e1, e0)

            # Choose actions ε-greedily among valid actions
            a0 = epsilon_greedy(Q0, s0, epsilon, valid0)
            a1 = epsilon_greedy(Q1, s1, epsilon, valid1)

            # Step the environment
            next_state, (r0, r1), done, _ = env.step(Action(a0), Action(a1))
            ne0, ne1 = next_state

            ns0 = state_to_idx(ne0, ne1)
            ns1 = state_to_idx(ne1, ne0)

            # Next-state valid masks
            next_valid0 = valid_actions_for_state(ne0, ne1)
            next_valid1 = valid_actions_for_state(ne1, ne0)

            # Q-learning updates, max only over valid next actions
            if done:
                max_next0 = 0.0
                max_next1 = 0.0
            else:
                max_next0 = np.max(Q0[ns0, next_valid0])
                max_next1 = np.max(Q1[ns1, next_valid1])

            target0 = r0 + gamma * max_next0
            Q0[s0, a0] += alpha * (target0 - Q0[s0, a0])

            target1 = r1 + gamma * max_next1
            Q1[s1, a1] += alpha * (target1 - Q1[s1, a1])

            state = next_state

    return Q0, Q1


# =========================
# 3. Policies (deterministic + stochastic)
# =========================

def greedy_policy_from_Q(Q):
    """
    Returns a function π(my_e, opp_e) -> Action
    that chooses argmax over Q, restricted to valid actions.
    """
    def policy(my_e, opp_e):
        s = state_to_idx(my_e, opp_e)
        valid = valid_actions_for_state(my_e, opp_e)
        valid_indices = np.where(valid)[0]
        q_vals = Q[s, valid_indices]
        best_idx = valid_indices[np.argmax(q_vals)]
        return Action(best_idx)
    return policy


def softmax(x, tau=1.0):
    """
    Compute softmax(x / tau) in a numerically stable way.
    tau = temperature; smaller -> more peaked distribution.
    """
    x = np.array(x, dtype=np.float64)
    z = (x - np.max(x)) / max(tau, 1e-8)
    e = np.exp(z)
    return e / np.sum(e)


def stochastic_policy_from_Q(Q, tau=0.3):
    """
    Returns a function pi(my_e, opp_e) -> prob vector over actions (len=NUM_ACTIONS)
    using softmax over Q-values with temperature tau,
    but with invalid actions forced to prob 0.
    """
    def policy(my_e, opp_e):
        s = state_to_idx(my_e, opp_e)
        valid = valid_actions_for_state(my_e, opp_e)
        q_s = Q[s]

        # Mask invalid actions by setting them very negative
        masked_q = np.full_like(q_s, -1e9, dtype=float)
        masked_q[valid] = q_s[valid]

        probs = softmax(masked_q, tau=tau)
        probs[~valid] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs
    return policy


def print_greedy_policy(Q, label="Greedy policy"):
    pi = greedy_policy_from_Q(Q)
    print(f"\n=== {label} ===")
    for my_e in range(NUM_ENERGY):
        for opp_e in range(NUM_ENERGY):
            a = pi(my_e, opp_e)
            print(f"State (my_e={my_e}, opp_e={opp_e}) -> {ACTION_NAMES[a]}")
        print()


def print_stochastic_policy(Q, label="Stochastic policy", tau=0.3):
    pi = stochastic_policy_from_Q(Q, tau=tau)
    print(f"\n=== {label} (tau={tau}) ===")
    for my_e in range(NUM_ENERGY):
        for opp_e in range(NUM_ENERGY):
            probs = pi(my_e, opp_e)
            print(f"State (my_e={my_e}, opp_e={opp_e}):")
            for a_idx, p in enumerate(probs):
                if p < 1e-3:
                    continue  # hide tiny probs for readability
                a = Action(a_idx)
                print(f"  {ACTION_NAMES[a]:>13}: {p:.3f}")
        print()


# =========================
# 4. Optional: quick test if run directly
# =========================

if __name__ == "__main__":
    Q0, Q1 = train_self_play(
        num_episodes=50000,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.01,
        seed=42,
        step_penalty=0.01,
    )

    Q_avg = 0.5 * (Q0 + Q1)

    print_greedy_policy(Q_avg, "Average symmetric greedy policy")
    print_stochastic_policy(Q_avg, "Average symmetric mixed policy", tau=0.2)

