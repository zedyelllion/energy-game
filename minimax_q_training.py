"""
Minimax-Q learning for the Energy Combat Game.

This uses an RL-style training loop (sampled episodes),
but the backup uses the minimax value of the stage game at each state.
It should converge much more reliably toward the Nash equilibrium
than plain self-play Q-learning.
"""

import numpy as np
from scipy.optimize import linprog

from energy_game_selfplay import (
    EnergyGameEnv,
    Action,
    NUM_ACTIONS,
    state_to_idx,
    valid_actions_for_state,
)

GAMMA = 0.9          # same as the rest of your project
ALPHA = 0.1
EPSILON = 0.1        # exploration for our actions
NUM_EPISODES = 200_000  # you can tune this


def solve_zero_sum_row_minimax(payoff_matrix):
    """
    Given matrix U (m x n) for row player,
    solve max_pi min_j sum_i pi_i * U[i,j].

    Returns:
        v  : value of the game for the row player
        pi : row player's optimal mixed strategy (length m)
    """
    U = payoff_matrix
    m, n = U.shape

    # variables: x = [pi_0,...,pi_{m-1}, v]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -v => maximize v

    # constraints: -sum_i pi_i U[i,j] + v <= 0 for all j
    A_ub = []
    b_ub = []
    for j in range(n):
        row = np.zeros(m + 1)
        row[:m] = -U[:, j]
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # equality: sum_i pi_i = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * m + [(None, None)]

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Minimax LP failed: {res.message}")

    x = res.x
    pi = x[:m]
    v = x[-1]

    # numerical cleanup
    pi = np.clip(pi, 0.0, 1.0)
    if pi.sum() > 0:
        pi /= pi.sum()

    return v, pi


class MinimaxQAgent:
    """
    Minimax-Q agent for the row player (player 0).
    We learn Q(s, a0, a1), value V(s), and our equilibrium policy pi(s, a0).
    The opponent during training can be simple (e.g., uniform over valid actions),
    since the minimax backup already guards against worst-case opponents.
    """

    def __init__(self, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # There are 3x3 = 9 (my_e, opp_e) states for nonterminal
        self.num_states = 9

        # Q[s_idx, a0, a1]
        self.Q = np.zeros((self.num_states, NUM_ACTIONS, NUM_ACTIONS), dtype=float)
        # state values
        self.V = np.zeros(self.num_states, dtype=float)
        # policy[s_idx, a0] (row player's mixed strategy)
        self.policy = np.zeros((self.num_states, NUM_ACTIONS), dtype=float)

    def get_state_index(self, state, player_id=0):
        """
        state is (e0, e1). From row player's perspective (player 0),
        state_to_idx should already map (my_e, opp_e) -> [0..8].
        If you later want a player-1 perspective, you'd swap.
        """
        return state_to_idx[state]

    def epsilon_greedy_action(self, s_idx, my_e, opp_e):
        """
        Sample our action a0 using epsilon-greedy from current minimax policy.
        If we don't yet have a meaningful policy at that state, fall back to uniform
        over valid actions.
        """
        valid = valid_actions_for_state(my_e, opp_e)
        valid_idxs = [a for a in range(NUM_ACTIONS) if valid[a]]

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_idxs)

        # use current minimax policy if non-zero; otherwise uniform
        probs = self.policy[s_idx].copy()
        probs = probs * valid
        if probs.sum() <= 0:
            probs = np.array(valid, dtype=float)
        probs /= probs.sum()
        return np.random.choice(np.arange(NUM_ACTIONS), p=probs)

    def sample_opponent_action(self, my_e, opp_e):
        """
        Simple training opponent: uniform over valid actions.
        This does NOT need to be optimal; minimax backup already
        computes robust values.
        You could also plug in a stronger fixed policy here.
        """
        valid = valid_actions_for_state(opp_e, my_e)  # note perspective swap
        valid_idxs = [a for a in range(NUM_ACTIONS) if valid[a]]
        return np.random.choice(valid_idxs)

    def minimax_backup_at_state(self, s_idx, my_e, opp_e):
        """
        Given current Q(s_idx, :, :), compute the stage-game payoff matrix
        restricted to valid actions, solve minimax, and update V[s_idx]
        and policy[s_idx].
        """
        # row-player valid actions (my perspective)
        valid_row = valid_actions_for_state(my_e, opp_e)
        row_actions = [a for a in range(NUM_ACTIONS) if valid_row[a]]
        # column-player valid actions (opponent perspective)
        valid_col = valid_actions_for_state(opp_e, my_e)
        col_actions = [b for b in range(NUM_ACTIONS) if valid_col[b]]

        if len(row_actions) == 0 or len(col_actions) == 0:
            self.V[s_idx] = 0.0
            self.policy[s_idx, :] = 0.0
            return

        # build small matrix U (m x n)
        U = np.zeros((len(row_actions), len(col_actions)), dtype=float)
        for i, a0 in enumerate(row_actions):
            for j, a1 in enumerate(col_actions):
                U[i, j] = self.Q[s_idx, a0, a1]

        v_s, pi_row = solve_zero_sum_row_minimax(U)

        # update V and policy
        self.V[s_idx] = v_s
        self.policy[s_idx, :] = 0.0
        for i, a0 in enumerate(row_actions):
            self.policy[s_idx, a0] = pi_row[i]

    def train(self, num_episodes=NUM_EPISODES, step_penalty=0.01, verbose=False):
        env = EnergyGameEnv(step_penalty=step_penalty)

        for ep in range(num_episodes):
            state = env.reset()  # e.g. returns (e0, e1)
            done = False

            while not done:
                e0, e1 = state
                s_idx = self.get_state_index(state, player_id=0)

                # choose actions
                a0 = self.epsilon_greedy_action(s_idx, e0, e1)
                a1 = self.sample_opponent_action(e0, e1)

                # step
                next_state, r0, r1, done = env.step(Action(a0), Action(a1))
                if done:
                    target = r0
                else:
                    s2_idx = self.get_state_index(next_state, player_id=0)
                    target = r0 + self.gamma * self.V[s2_idx]

                # Q update
                old_q = self.Q[s_idx, a0, a1]
                self.Q[s_idx, a0, a1] = old_q + ALPHA * (target - old_q)

                # minimax backup at current state (update V & policy)
                self.minimax_backup_at_state(s_idx, e0, e1)

                state = next_state

            if verbose and (ep + 1) % 20000 == 0:
                print(f"[Minimax-Q] Episode {ep+1}/{num_episodes}")

        if verbose:
            print("Minimax-Q training done.")

    def get_policy_for_state(self, my_e, opp_e):
        s_idx = self.get_state_index((my_e, opp_e))
        return self.policy[s_idx].copy()


if __name__ == "__main__":
    agent = MinimaxQAgent()
    agent.train(num_episodes=200_000, step_penalty=0.01, verbose=True)

    # Show policy for all states (my_e, opp_e)
    for my_e in range(3):
        for opp_e in range(3):
            pi = agent.get_policy_for_state(my_e, opp_e)
            print(f"State (my_e={my_e}, opp_e={opp_e}):")
            for a_idx, p in enumerate(pi):
                if p > 1e-4:
                    print(f"  {Action(a_idx).name:>13}: {p:.3f}")
            print()
        print("-" * 40)

