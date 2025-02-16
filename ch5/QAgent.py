import random
import numpy as np


class QAgent():

    def __init__(self):
        self.q_table = np.zeros((5,7,4)) # q_value를 저장하는 변수
        self.eps = 0.9
        self.alpha = 0.01

    def select_action(self,s):
        # eps-greedy로 액션을 선택
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, history):

        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x, y = s
            self.q_table[x, y, a] = self.q_table[x, y, a] + self.alpha * (cum_reward - self.q_table[x, y, a])
            cum_reward = cum_reward + r

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 어느 액션의 q 값이 가장 높았는가?
        q_lst = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len (row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action

        print(data)


class SAgent():

    def __init__(self):
        self.q_table = np.zeros((5, 7, 4))  # q_value를 저장하는 변수
        self.eps = 0.9

    def select_action(self, s):
        # eps-greedy로 액션을 선택
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            action_val = self.q_table[x, y, :]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        a_prime = self.select_action(s_prime)
        self.q_table[x,y,a] = self.q_table[x,y,a] + 0.1* (r + self.q_table[next_x,next_y,a_prime] - self.q_table[x,y,a])

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 어느 액션의 q 값이 가장 높았는가?
        q_lst = self.q_table.tolist()
        data = np.zeros((5, 7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action

        print(data)


class QLAgent():

    def __init__(self):
        self.q_table = np.zeros((5,7,4))
        self.eps = 0.9

    def select_action(self, s):
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,3)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        # Q러닝 업데이트 식을 사용함
        self.q_table[x,y,a] = self.q_table[x,y,a] + 0.1 * (r + np.amax(self.q_table[next_x,next_y,:]) - self.q_table[x,y,a])

    def anneal_eps(self):
        self.eps -=0.01 # DECAYING EPSILON : 탐험을 갈수록 적게 하도록
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        q_lst = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)