# eval_mixed_vs_greedy.py

import numpy as np
import matplotlib.pyplot as plt

from energy_game_selfplay import (
    Action,
    EnergyGameEnv,
    NUM_ACTIONS,
    train_self_play,
    greedy_policy_from_Q,
    stochastic_policy_from_Q,
)


def run_episode(env, policy_A, policy_B, max_steps=50):
    """
    One episode of:
      - Player A: mixed strategy (stochastic policy)
      - Player B: greedy strategy

    Returns:
      outcome: +1 if A wins, -1 if A loses, 0 if draw / max_steps reached
      total_r0, total_r1: cumulative rewards
    """
    state = env.reset()
    done = False
    steps = 0

    total_r0 = 0.0
    total_r1 = 0.0

    while not done and steps < max_steps:
        e0, e1 = state

        # A is player 0: sees (my_e=e0, opp_e=e1)
        probs_A = policy_A(e0, e1)  # vector length NUM_ACTIONS
        probs_A = np.asarray(probs_A, dtype=float)
        probs_A = probs_A / probs_A.sum()  # normalize just in case

        a0_idx = int(np.random.choice(np.arange(NUM_ACTIONS), p=probs_A))
        a0 = Action(a0_idx)

        # B is player 1: sees (my_e=e1, opp_e=e0)
        a1 = policy_B(e1, e0)  # returns an Action

        next_state, (r0, r1), done, _ = env.step(a0, a1)
        total_r0 += r0
        total_r1 += r1

        state = next_state
        steps += 1

    # Decide outcome from cumulative rewards
    if total_r0 > total_r1:
        outcome = 1   # A wins
    elif total_r0 < total_r1:
        outcome = -1  # A loses
    else:
        outcome = 0   # tie / max_steps reached

    return outcome, total_r0, total_r1


def evaluate_mixed_vs_greedy(
    num_train_episodes=50000,
    num_eval_episodes=5000,
    step_penalty=0.01,
    tau_A=0.2,
    seed=123,
):
    np.random.seed(seed)

    # 1. Train self-play Q-tables
    Q0, Q1 = train_self_play(
        num_episodes=num_train_episodes,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.01,
        seed=seed,
        step_penalty=step_penalty,
    )

    Q_avg = 0.5 * (Q0 + Q1)

    # 2. Build policies
    # A: mixed strategy (stochastic) from Q_avg
    policy_A = stochastic_policy_from_Q(Q_avg, tau=tau_A)
    # B: greedy best-response from same Q_avg
    policy_B = greedy_policy_from_Q(Q_avg)

    # 3. Evaluate
    env = EnergyGameEnv(step_penalty=step_penalty)

    outcomes = []  # +1 / 0 / -1 for each episode
    for ep in range(num_eval_episodes):
        outcome, r0, r1 = run_episode(env, policy_A, policy_B, max_steps=50)
        outcomes.append(outcome)

    outcomes = np.array(outcomes)
    wins_A = (outcomes == 1).astype(float)
    draws = (outcomes == 0).astype(float)
    losses_A = (outcomes == -1).astype(float)

    # Cumulative rates
    denom = np.arange(1, num_eval_episodes + 1)
    cum_win_rate_A = np.cumsum(wins_A) / denom
    cum_draw_rate = np.cumsum(draws) / denom
    cum_loss_rate_A = np.cumsum(losses_A) / denom

    return {
        "outcomes": outcomes,
        "cum_win_rate_A": cum_win_rate_A,
        "cum_draw_rate": cum_draw_rate,
        "cum_loss_rate_A": cum_loss_rate_A,
    }


def plot_results(cum_win_rate_A, cum_draw_rate, cum_loss_rate_A):
    episodes = np.arange(1, len(cum_win_rate_A) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, cum_win_rate_A, label="A win rate (mixed vs greedy)")
    plt.plot(episodes, cum_draw_rate, label="Draw rate")
    plt.plot(episodes, cum_loss_rate_A, label="A loss rate")

    plt.xlabel("Evaluation episode")
    plt.ylabel("Cumulative rate")
    plt.title("Agent A (mixed) vs Agent B (greedy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results = evaluate_mixed_vs_greedy(
        num_train_episodes=50000,  # training for Q
        num_eval_episodes=5000,    # evaluation games
        step_penalty=0.01,
        tau_A=0.2,
        seed=123,
    )

    print(f"Final A win rate:  {results['cum_win_rate_A'][-1]:.3f}")
    print(f"Final draw rate:   {results['cum_draw_rate'][-1]:.3f}")
    print(f"Final A loss rate: {results['cum_loss_rate_A'][-1]:.3f}")

    plot_results(
        results["cum_win_rate_A"],
        results["cum_draw_rate"],
        results["cum_loss_rate_A"],
    )

