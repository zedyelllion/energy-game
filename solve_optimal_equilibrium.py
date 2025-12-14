# solve_optimal_equilibrium.py
#
# 使用动态规划 + 线性规划，求这个零和 Markov 游戏的
# 近似最优混合策略（折扣纳什均衡），并用表格和图表展示结果。

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import pandas as pd

from energy_game_selfplay import (
    Action,
    EnergyGameEnv,
    NUM_ACTIONS,
    state_to_idx,
    valid_actions_for_state,
)

# 所有状态 (my_e, opp_e)
ALL_STATES = [(i, j) for i in range(3) for j in range(3)]
STATE_INDEX = {s: k for k, s in enumerate(ALL_STATES)}
NUM_STATES = len(ALL_STATES)

GAMMA = 0.9          # 折扣因子，和 Q-learning 里保持一致
STEP_PENALTY = 0.01  # 每一步的小惩罚，也和环境保持一致


def one_step(state, a_row, a_col, step_penalty=STEP_PENALTY):
    """
    用 EnergyGameEnv 做一步转移，保证和训练时完全一致。
    row-player = A, col-player = B.
    返回：next_state, reward_for_A, done
    """
    env = EnergyGameEnv(step_penalty=step_penalty)
    env.state = state
    env.done = False
    next_state, (rA, rB), done, _ = env.step(Action(a_row), Action(a_col))
    return next_state, rA, done


def build_payoff_matrix(state, V):
    """
    给定当前价值函数 V (dict: state -> value),
    在某个状态 s 下构造矩阵博弈 U(s)，维度 m x n：
      m = A 的合法动作数
      n = B 的合法动作数
    U[i,j] = r(s, a_i, b_j) + gamma * V(s')
    """
    my_e, opp_e = state
    acts_A = np.array([a for a in range(NUM_ACTIONS)
                       if valid_actions_for_state(my_e, opp_e)[a]])
    acts_B = np.array([b for b in range(NUM_ACTIONS)
                       if valid_actions_for_state(opp_e, my_e)[b]])

    m = len(acts_A)
    n = len(acts_B)
    U = np.zeros((m, n), dtype=float)

    for i, aA in enumerate(acts_A):
        for j, aB in enumerate(acts_B):
            s2, rA, done = one_step(state, aA, aB, STEP_PENALTY)
            cont = 0.0 if done else GAMMA * V[s2]
            U[i, j] = rA + cont

    return acts_A, acts_B, U


def solve_zero_sum_row_player(U):
    """
    对矩阵博弈 U (m x n) 求 row-player 的 max-min 策略 π 和对应价值 v。
    线性规划形式：
        maximize v
        s.t.   sum_i π_i = 1, π_i >= 0
               for all j: sum_i π_i U[i,j] >= v
    linprog 是求 min，所以我们最小化 -v。
    """
    m, n = U.shape
    # 变量 x = [π_0, ..., π_{m-1}, v]
    c = np.zeros(m + 1)
    c[-1] = -1.0  # minimize -v <=> maximize v

    # 不等式约束: -sum_i π_i U[i,j] + v <= 0
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

    # 等式: sum_i π_i = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # 变量范围: π_i ∈ [0,1], v 无界
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
        raise RuntimeError(f"linprog failed: {res.message}")

    x = res.x
    pi = x[:m]
    v = x[-1]
    # 数值误差修一下
    pi = np.clip(pi, 0.0, 1.0)
    if pi.sum() > 0:
        pi = pi / pi.sum()
    return v, pi


def value_iteration_zero_sum(max_iters=500, tol=1e-6):
    """
    Shapley value iteration:
      V_{k+1}(s) = val( U(s; V_k) )
    并记录每个状态的最优混合策略 π*_A(s, ·)。
    """
    # V 初始为 0
    V = {s: 0.0 for s in ALL_STATES}
    policy = {s: None for s in ALL_STATES}

    for it in range(max_iters):
        delta = 0.0
        V_new = {}

        for s in ALL_STATES:
            acts_A, acts_B, U = build_payoff_matrix(s, V)
            v_s, pi_s = solve_zero_sum_row_player(U)
            V_new[s] = v_s

            # 记录策略：映射 action -> prob
            probs = {int(acts_A[i]): pi_s[i] for i in range(len(acts_A))}
            policy[s] = probs

            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        print(f"Iter {it+1}, max value diff = {delta:.6e}")
        if delta < tol:
            print("Converged.")
            break

    return V, policy


# ========= 表格展示：把策略和价值转成 DataFrame ========== #

def policy_to_dataframe(policy, V):
    """
    把 policy (state -> {action: prob}) 和 V(s)
    整理成一个 pandas DataFrame：
    列：my_e, opp_e, V,
        p_STORE, p_ATTACK_SMALL, p_ATTACK_BIG, p_DEFEND_SMALL, p_DEFEND_BIG
    """
    rows = []
    for my_e in range(3):
        for opp_e in range(3):
            s = (my_e, opp_e)
            probs_dict = policy[s]  # {action_idx: prob}
            row = {
                "my_e": my_e,
                "opp_e": opp_e,
                "V": V[s],
                "p_STORE": 0.0,
                "p_ATTACK_SMALL": 0.0,
                "p_ATTACK_BIG": 0.0,
                "p_DEFEND_SMALL": 0.0,
                "p_DEFEND_BIG": 0.0,
            }
            for a_idx, p in probs_dict.items():
                a = Action(a_idx)
                if a == Action.STORE:
                    row["p_STORE"] = p
                elif a == Action.ATTACK_SMALL:
                    row["p_ATTACK_SMALL"] = p
                elif a == Action.ATTACK_BIG:
                    row["p_ATTACK_BIG"] = p
                elif a == Action.DEFEND_SMALL:
                    row["p_DEFEND_SMALL"] = p
                elif a == Action.DEFEND_BIG:
                    row["p_DEFEND_BIG"] = p
            rows.append(row)

    df = pd.DataFrame(rows)
    # 按 my_e, opp_e 排个序，方便看
    df = df.sort_values(by=["my_e", "opp_e"]).reset_index(drop=True)
    return df


# ========= 图表展示 1：V(s) 热力图 ========== #

def plot_value_heatmap(V):
    """
    画一个 3x3 的热力图，显示不同 (my_e, opp_e) 下的 V(s)。
    横轴：opp_e, 纵轴：my_e
    """
    value_grid = np.zeros((3, 3))
    for my_e in range(3):
        for opp_e in range(3):
            value_grid[my_e, opp_e] = V[(my_e, opp_e)]

    plt.figure(figsize=(5, 4))
    im = plt.imshow(value_grid, origin="upper", cmap="coolwarm")
    plt.colorbar(im, label="V(s) (expected value for row player)")
    plt.xticks([0, 1, 2], ["opp_e=0", "opp_e=1", "opp_e=2"])
    plt.yticks([0, 1, 2], ["my_e=0", "my_e=1", "my_e=2"])

    # 在格子中标注数值
    for i in range(3):
        for j in range(3):
            v = value_grid[i, j]
            plt.text(
                j, i, f"{v:.2f}",
                ha="center", va="center", color="black", fontsize=10
            )

    plt.title("State Value Heatmap V(my_e, opp_e)")
    plt.tight_layout()
    plt.show()


# ========= 图表展示 2：每个状态的策略条形图（3x3 子图） ========== #

def plot_policy_grid(policy):
    """
    画一个 3x3 的子图网格，每个子图对应一个状态 (my_e, opp_e)，
    显示该状态下 5 个动作的概率柱状图。
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    action_labels = [
        "STORE", "A_SMALL", "A_BIG", "D_SMALL", "D_BIG"
    ]

    for my_e in range(3):
        for opp_e in range(3):
            ax = axes[my_e, opp_e]
            s = (my_e, opp_e)
            probs_dict = policy[s]

            probs = np.zeros(NUM_ACTIONS)
            for a_idx, p in probs_dict.items():
                probs[int(a_idx)] = p

            ax.bar(range(NUM_ACTIONS), probs)
            ax.set_ylim(0, 1)
            ax.set_xticks(range(NUM_ACTIONS))
            ax.set_xticklabels(action_labels, rotation=45, fontsize=8)
            ax.set_title(f"(my_e={my_e}, opp_e={opp_e})", fontsize=10)

    plt.suptitle("Optimal Mixed Strategy per State", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ========= 原来的文本打印函数（可以保留也可以不用） ========== #

def pretty_print_policy(policy, title="Optimal mixed strategy (row player)"):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for my_e in range(3):
        for opp_e in range(3):
            s = (my_e, opp_e)
            probs = policy[s]
            print(f"State (my_e={my_e}, opp_e={opp_e}):")
            for a_idx, p in sorted(probs.items()):
                if p < 1e-4:
                    continue
                name = Action(a_idx).name
                print(f"  {name:>13}: {p:.3f}")
            print()
        print("-" * 40)


# ========= 主程序：求解 + 表格 + 图表 ========== #

if __name__ == "__main__":
    V, policy = value_iteration_zero_sum(max_iters=500, tol=1e-6)

    print("\nValue at initial state (0,0):", V[(0, 0)])

    # 文本方式打印（可选）
    pretty_print_policy(policy)

    # 1. 用 DataFrame 展示并另存为 CSV
    df_policy = policy_to_dataframe(policy, V)
    print("\n=== Optimal Policy Table ===")
    print(df_policy)

    # 如果你想保存成文件：
    df_policy.to_csv("optimal_policy_table.csv", index=False)
    print("\n已保存到 optimal_policy_table.csv")

    # 2. 画 V(s) 热力图
    plot_value_heatmap(V)

    # 3. 画 3x3 策略条形图
    plot_policy_grid(policy)

