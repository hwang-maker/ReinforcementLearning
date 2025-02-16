import numpy as np

class MarkovDecisionProcess:
    def __init__(self, states, actions, transition_prob, rewards, gamma=0.9, theta=1e-6):
        self.states = states
        self.actions = actions
        self.P = transition_prob
        self.R = rewards
        self.gamma = gamma
        self.theta = theta
        self.V = {s: 0 for s in states}  # 가치 함수 초기화
        self.policy = {s: np.random.choice(actions) if actions else "None" for s in states}  # 종료 상태 예외 처리

    def value_iteration(self):
        while True:
            delta = 0
            new_V = self.V.copy()

            for s in self.states:
                q_values = []
                for a in self.actions:
                    if a in self.P[s]:  # 해당 상태에서 가능한 행동인지 확인
                        q_value = sum(
                            self.P[s][a].get(s_next, 0) * (self.R[s].get(a, 0) + self.gamma * self.V[s_next])
                            for s_next in self.states
                        )
                        q_values.append(q_value)

                if q_values:  # 행동이 존재하는 경우에만 업데이트
                    new_V[s] = max(q_values)
                delta = max(delta, abs(self.V[s] - new_V[s]))

            self.V = new_V
            if delta < self.theta:
                break

        self.update_policy()

    def update_policy(self):
        for s in self.states:
            best_action = None
            best_value = float("-inf")

            for a in self.actions:
                if a in self.P[s]:  # 해당 상태에서 가능한 행동인지 확인
                    q_value = sum(
                        self.P[s][a].get(s_next, 0) * (self.R[s].get(a, 0) + self.gamma * self.V[s_next])
                        for s_next in self.states
                    )

                    if q_value > best_value:
                        best_value = q_value
                        best_action = a

            self.policy[s] = best_action if best_action else "None"

    def display_results(self):
        print("\n[최적 가치 함수]")
        for s in self.states:
            print(f"V({s}) = {self.V[s]:.4f}")

        print("\n[최적 정책]")
        for s in self.states:
            print(f"π({s}) = {self.policy[s]}")

# -----------------------------
#  예제 환경 정의 (오류 수정)
# -----------------------------
states = ["S1", "S2", "S3", "S4", "S5", "S6"]
actions = ["facebook", "quit", "study", "sleep", "pub"]

transition_prob = {
    "S1": {"facebook": {"S1": 1.0}, "quit": {"S2": 1.0}},
    "S2": {"facebook": {"S1": 1.0}, "study": {"S3": 1.0}, "quit": {"S2": 0.0}},  # quit 추가
    "S3": {"study": {"S4": 1.0}, "sleep": {"S5": 1.0}, "quit": {"S3": 0.0}},  # quit 추가
    "S4": {"study": {"S5": 1.0}, "pub": {"S6": 1.0}, "quit": {"S4": 0.0}},  # quit 추가
    "S5": {"None": {"S5": 1.0}},  # 종료 상태
    "S6": {"None": {"S6": 1.0}},  # 종료 상태
}

rewards = {
    "S1": {"facebook": 5, "quit": 10},
    "S2": {"facebook": -1, "study": 2, "quit": 0},  # quit 추가
    "S3": {"study": 0, "sleep": 3, "quit": 0},  # quit 추가
    "S4": {"study": 1, "pub": 4, "quit": 0},  # quit 추가
    "S5": {"None": 0},
    "S6": {"None": 0},  # 종료 상태
}

# MDP 객체 생성
mdp = MarkovDecisionProcess(states, actions, transition_prob, rewards)

# 가치 반복 수행
mdp.value_iteration()

# 결과 출력
mdp.display_results()

