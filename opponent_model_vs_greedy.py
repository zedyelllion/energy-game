# opponent_model_vs_greedy.py

import numpy as np
import matplotlib.pyplot as plt

from energy_game_selfplay import (
    Action,
    EnergyGameEnv,
    NUM_ACTIONS,
    NUM_STATES,
    state_to_idx,
    valid_actions_for_state,
    train_self_play,
    greedy_policy_from_Q,
)


# ---------------- Opponent Model ---------------- #

class OpponentModel:
    """
    简单对手模型：统计在每个状态 s 下，对手 B 选择各个动作的次数，
    得到经验分布 P(a_B | s)。
    """

    def __init__(self, num_states, num_actions):
        # 加 1 做 Dirichlet(1) 先验，避免一开始全是 0
        self.counts = np.ones((num_states, num_actions), dtype=float)

    def update(self, s_idx, a_idx):
        self.counts[s_idx, a_idx] += 1.0

    def get_probs(self, s_idx):
        c = self.counts[s_idx]
        return c / c.sum()


# ------------- 辅助：ε-greedy 选动作（只看 A 的） ------------- #

def epsilon_greedy_A(Q_A, s_idx, epsilon, valid_mask):
    valid_indices = np.where(valid_mask)[0]
    if np.random.rand() < epsilon:
        return int(np.random.choice(valid_indices))
    q_vals = Q_A[s_idx, valid_indices]
    best = valid_indices[np.argmax(q_vals)]
    return int(best)


# ------------- 训练：A 对抗固定 greedy-B + 建立对手模型 ------------- #

def train_best_response_with_opponent_model(
    policy_B,
    num_episodes=50000,
    alpha=0.1,
    gamma=0.9,
    epsilon_start=0.2,
    epsilon_end=0.01,
    step_penalty=0.01,
    seed=0,
):
    """
    训练一个只学习 A 的 Q 表：
      - 环境中 B 永远使用固定 greedy 策略 policy_B
      - A 用 Q-learning 学 best response
      - 同时用 OpponentModel 记录 B 的行为分布
    """
    np.random.seed(seed)

    env = EnergyGameEnv(step_penalty=step_penalty)

    Q_A = np.zeros((NUM_STATES, NUM_ACTIONS))
    opp_model = OpponentModel(NUM_STATES, NUM_ACTIONS)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        frac = episode / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        while not done:
            eA, eB = state  # A 是 player 0

            s_idx = state_to_idx(eA, eB)
            validA = valid_actions_for_state(eA, eB)

            # A: ε-greedy 选动作
            aA_idx = epsilon_greedy_A(Q_A, s_idx, epsilon, validA)
            aA = Action(aA_idx)

            # B: 固定 greedy 策略
            aB = policy_B(eB, eA)
            aB_idx = int(aB)

            # 更新对手模型
            opp_model.update(s_idx, aB_idx)

            # 与环境交互
            next_state, (rA, rB), done, _ = env.step(aA, aB)
            eA2, eB2 = next_state
            s2_idx = state_to_idx(eA2, eB2)

            validA_next = valid_actions_for_state(eA2, eB2)

            if done:
                max_next = 0.0
            else:
                max_next = np.max(Q_A[s2_idx, validA_next])

            target = rA + gamma * max_next
            Q_A[s_idx, aA_idx] += alpha * (target - Q_A[s_idx, aA_idx])

            state = next_state

    return Q_A, opp_model


# ------------- 评估：A(greedy) vs B(greedy) ------------- #

def run_episode(env, policy_A, policy_B, max_steps=50):
    state = env.reset()
    done = False
    steps = 0

    total_rA = 0.0
    total_rB = 0.0

    while not done and steps < max_steps:
        eA, eB = state

        aA = policy_A(eA, eB)      # A: greedy w.r.t. Q_A
        aB = policy_B(eB, eA)      # B: greedy w.r.t. Q_avg

        next_state, (rA, rB), done, _ = env.step(aA, aB)
        total_rA += rA
        total_rB += rB

        state = next_state
        steps += 1

    if total_rA > total_rB:
        outcome = 1   # A wins
    elif total_rA < total_rB:
        outcome = -1  # A loses
    else:
        outcome = 0   # draw
    return outcome, total_rA, total_rB


def evaluate_best_response_vs_greedy(
    num_train_selfplay=50000,
    num_train_BR=50000,
    num_eval_episodes=5000,
    step_penalty=0.01,
    seed=123,
):
    np.random.seed(seed)

    # 1. 先用自博弈得到“均衡策略” Q_avg，用来定义 B 的 greedy 策略
    Q0, Q1 = train_self_play(
        num_episodes=num_train_selfplay,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.01,
        seed=seed,
        step_penalty=step_penalty,
    )
    Q_avg = 0.5 * (Q0 + Q1)
    policy_B = greedy_policy_from_Q(Q_avg)

    # 2. 在固定 greedy-B 上训练 A + 对手模型
    Q_A, opp_model = train_best_response_with_opponent_model(
        policy_B=policy_B,
        num_episodes=num_train_BR,
        alpha=0.1,
        gamma=0.9,
        epsilon_start=0.2,
        epsilon_end=0.01,
        step_penalty=step_penalty,
        seed=seed + 1,
    )

    # A 自己也用 greedy 策略来打（真正对抗时不再 epsilon）
    def policy_A_greedy(eA, eB):
        s_idx = state_to_idx(eA, eB)
        validA = valid_actions_for_state(eA, eB)
        valid_idx = np.where(validA)[0]
        q_vals = Q_A[s_idx, valid_idx]
        best = valid_idx[np.argmax(q_vals)]
        return Action(best)

    # 3. 评估
    env = EnergyGameEnv(step_penalty=step_penalty)

    outcomes = []
    for ep in range(num_eval_episodes):
        outcome, rA, rB = run_episode(env, policy_A_greedy, policy_B, max_steps=50)
        outcomes.append(outcome)

    outcomes = np.array(outcomes)
    wins_A = (outcomes == 1).astype(float)
    draws = (outcomes == 0).astype(float)
    losses_A = (outcomes == -1).astype(float)

    denom = np.arange(1, num_eval_episodes + 1)
    cum_win_rate_A = np.cumsum(wins_A) / denom
    cum_draw_rate = np.cumsum(draws) / denom
    cum_loss_rate_A = np.cumsum(losses_A) / denom

    return cum_win_rate_A, cum_draw_rate, cum_loss_rate_A, opp_model


def plot_results(cum_win_rate_A, cum_draw_rate, cum_loss_rate_A, title):
    episodes = np.arange(1, len(cum_win_rate_A) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, cum_win_rate_A, label="A win rate")
    plt.plot(episodes, cum_draw_rate, label="Draw rate")
    plt.plot(episodes, cum_loss_rate_A, label="A loss rate")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Cumulative rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cum_win, cum_draw, cum_loss, opp_model = evaluate_best_response_vs_greedy(
        num_train_selfplay=50000,  # 先训练均衡策略 (给 B 用)
        num_train_BR=50000,        # 再训练 A 的 best response
        num_eval_episodes=5000,
        step_penalty=0.01,
        seed=123,
    )

    print(f"Final A win rate vs greedy-B:  {cum_win[-1]:.3f}")
    print(f"Final draw rate:               {cum_draw[-1]:.3f}")
    print(f"Final A loss rate:             {cum_loss[-1]:.3f}")

    plot_results(cum_win, cum_draw, cum_loss,
                 "Agent A (best-response with opponent model) vs Greedy Agent B")

