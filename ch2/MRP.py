import numpy as np


class MarkovRewardProcess:
    def __init__(self, states, transition_matrix, rewards, gamma=0.9):
        """
        :param states: 상태 리스트
        :param transition_matrix: 상태 전이 확률 행렬 (N*N)
        :param rewards: 보상 벡터 (N,)
        :param gamma: DISCOUNT FACTOR(0 <= GAMMA <=1)
        """
        self.states = states
        self.transition_matrix = transition_matrix
        self.rewards = rewards
        self.gamma = gamma
        self.value_function = np.zeros(len(states))
    ''' 넘파이로 구현해보기..? '''
    def compute_value_function(self, tol=1e-6):
        "벨만 방정식을 사용한 state-value function 계산"
        V = np.zeros(len(self.states))
        while True:
            V_new = self.rewards + self.gamma * np.dot(self.transition_matrix,V) # 0.9 * np.dot(self.transition_matrix,V) 하면 안되더라!
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new
        self.value_function = V
        return V

    def simulate(self,start_state, num_steps=10):
        "MRP 시뮬레이션을 실행하여 상태와 누적보상 반환"

        state_idx = self.states.index(start_state)
        total_reward = 0
        discount = 1.0
        trajectory = [self.states[state_idx]]

        for _ in range(num_steps):
            reward = self.rewards[state_idx]
            total_reward += discount * reward
            discount *= self.gamma
            next_state_idx = np.random.choice(len(self.states), p=self.transition_matrix[state_idx])
            state_idx = next_state_idx
            trajectory.append(self.states[state_idx])
        return trajectory, total_reward


state = ['Facebook', 'Class1', 'Class2', 'Class3', 'Sleep', 'Pass','Pub']

# State = ['Facebook', 'Class1', 'Class2', 'Class3', 'Sleep', 'Pass','Pub']

transition_matrix = [
    [.9,.1,0.,0.,0.,0.,0.], # Facebook
    [.5,0.,.5,0.,0.,0.,0.], # Class1
    [0.,0.,0.,.8,.2,0.,0.], # Class2
    [0.,0.,0.,0.,0.,.6,.4], # Class3
    [0.,0.,0.,0.,1.,0.,0.], # Sleep
    [0.,0.,0.,0.,1,0.,0.], # Pass
    [0.,.2,.4,.4,0.,0.,0.] # Pub
]

# State = ['Facebook', 'Class1', 'Class2', 'Class3', 'Sleep', 'Pass','Pub']

rewards = [-1,-2,-2,-2,0,10,1]

mrp = MarkovRewardProcess(state, transition_matrix, rewards)

value_function = mrp.compute_value_function()
print("상태 가치 함수(v)", value_function)

# 시뮬레이션
start_state = 'Class3'
trajectory, total_reward = mrp.simulate(start_state)

print('시뮬레이션 결과:',trajectory)
print('총 보상: ',total_reward)