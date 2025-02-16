import numpy as np

def epsilon_greedy(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # 탐색 (랜덤 행동)
    else:
        return np.argmax(Q[state, :])  # 활용 (최적 행동 선택)

# Q-테이블 초기화 (5x4)
Q = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.2, 0.8, 0.1],  # 현재 상태 S2
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

epsilon = 0.2
n_actions = 4

state = 2  # 현재 상태 S2



for i in range(20):
    # epsilon-greedy로 행동 선택
    chosen_action = epsilon_greedy(Q, state, epsilon, n_actions)

    print(f"선택된 행동: {chosen_action} (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT)")


