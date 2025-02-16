import gym
from gym import spaces
import numpy as np


class GridworldEnv(gym.Env):
    """
    Gridworld Environment with policy evaluation support.

    Args:
        grid_size: Tuple representing the size of the grid (default: (4, 4)).
        policy: [S, A] shaped matrix representing the policy.
        theta: Convergence threshold for value function evaluation.
        discount_factor: Discount factor (gamma) for future rewards.
    """
    reward_range = (-1, 0)

    def __init__(self, grid_size=(4, 4), policy=None, theta=1e-6, discount_factor=0.9):
        super(GridworldEnv, self).__init__()

        self.grid_size = grid_size
        self.observation_space = spaces.Discrete(grid_size[0] * grid_size[1])  # S
        self.action_space = spaces.Discrete(4)  # A (상, 우, 하, 좌)

        self.policy = policy if policy is not None else np.ones(
            (self.observation_space.n, self.action_space.n)) / self.action_space.n
        self.theta = theta
        self.discount_factor = discount_factor

        self.gridworld = np.arange(self.observation_space.n).reshape(grid_size)
        self.gridworld[-1, -1] = 0  # 종료 상태 (목표 지점)

        # 상태 전이 행렬 초기화
        self.P = np.zeros((self.action_space.n, self.observation_space.n, self.observation_space.n))
        self.P[:, 0, 0] = 1  # 종료 상태에서는 모든 액션이 자기 자신을 가리킴



        # 상태 전이 정의 ex) state5, action(우회전) 선택시 -> state6로 이동
        for s in self.gridworld.flat[1:-1]:  # 종료 상태(0) 제외
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(range(self.action_space.n), [(-1, 0), (0, 1), (1, 0), (0, -1)]):  # 상, 우, 하, 좌
                next_row = max(0, min(row + d[0], self.grid_size[0] - 1))
                next_col = max(0, min(col + d[1], self.grid_size[1] - 1))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        # 보상 행렬 정의
        self.R = np.full((self.action_space.n, self.observation_space.n), -1)
        self.R[:, 0] = 0  # 종료 상태의 보상은 0

    def reset(self):
        """환경을 초기 상태로 리셋하고, 초기 상태 반환"""
        self.current_state = 1  # 종료 상태(0) 제외하고 시작
        return self.current_state

    def step(self, action):
        """주어진 액션을 수행하고, 다음 상태, 보상, 종료 여부 반환"""
        next_state_probs = self.P[action, self.current_state]
        next_state = np.random.choice(np.arange(self.observation_space.n), p=next_state_probs)  # 가능한 다음 상태 선택

        reward = self.R[action, self.current_state]
        self.current_state = next_state

        done = self.current_state == 0  # 종료 상태 여부
        return next_state, reward, done, {}

    def render(self):
        """환경을 시각적으로 출력"""

        if self.current_state not in self.gridworld:
            print('Error: current_state is not in the grid!')
            return
        grid = np.full(self.grid_size, " . ")
        row, col = np.argwhere(self.gridworld == self.current_state)[0]
        grid[row, col] = " A "  # 에이전트 위치
        grid[self.grid_size[0] - 1, self.grid_size[1] - 1] = " G "  # 목표 지점

        print("\n".join(["".join(row) for row in grid]))
        print()

    def evaluate_policy(self):
        """현재 정책에 대한 가치 함수를 계산"""
        V = np.zeros(self.observation_space.n)

        while True:
            delta = 0  # 최대 변화량 추적
            for s in range(self.observation_space.n):
                v = 0
                for a, action_prob in enumerate(self.policy[s]):
                    for next_state in range(self.observation_space.n):
                        prob = self.P[a, s, next_state]
                        reward = self.R[a, s]

                        # 벨만 방정식 적용
                        v += action_prob * prob * (reward + self.discount_factor * V[next_state])

                delta = max(delta, abs(V[s] - v))
                V[s] = v

            if delta < self.theta:
                break  # 충분히 수렴하면 종료

        return V

    def one_step_lookahead(self,state,V):

        A = np.zeros(self.action_space.n)
        for a in range(self.action_space.n):
            for next_state in range(self.observation_space.n):
                prob = self.P[a,state,next_state]
                reward = self.R[a,state]
                A[a] += prob * (reward + self.discount_factor * V[next_state])

        return A

    def policy_improvement(self):
        V = self.evaluate_policy()
        policy_stable = True
        new_policy = np.copy(self.policy)  # 기존 정책 복사

        for s in range(self.observation_space.n):
            chosen_a = np.argmax(self.policy[s])  # 현재 정책에서 선택한 행동
            action_values = self.one_step_lookahead(s, V)
            best_a = np.argmax(action_values)  # 최적 행동 찾기

            if chosen_a != best_a:
                policy_stable = False  # 정책이 변경됨

            new_policy[s] = np.eye(self.action_space.n)[best_a]  # One-hot Encoding

        return new_policy, V


