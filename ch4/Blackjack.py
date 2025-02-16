import gym
import numpy as np
from collections import defaultdict


env = gym.make('Blackjack-v1')

def policy(state):
    player_sum, dealer_card, usable_ace = state
    return 0 if player_sum>=20 else 1 # 0:Stand 1: Hit

def first_visit_mc(env, policy, num_episodes=50000, gamma = 1.0):
    value_table = defaultdict(float)
    returns =defaultdict(list)

    for _ in range(num_episodes):
        episode = []
        state = env.reset()[0]
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _ ,_ = env.step(action)
            episode.append((state,reward))
            state = next_state

        G = 0
        visited_states = set()

        for state, reward in reversed(episode):
            G = reward + gamma * G
            if state not in visited_states:
                visited_states.add(state)
                returns[state].append(G)
                value_table[state] = np.mean(returns[state])

    return value_table


def every_visit_mc (env,policy, num_episodes=50000, gamma=1.0):
    value_table = defaultdict(float)
    returns = defaultdict(list)

    for _ in range(num_episodes):
        episode = []
        state = env.reset()[0]
        done = False

        while not done:
            action = policy(state)
            next_state, reward, done, _ ,_ = env.step(action)
            episode.append((state, reward))
            state = next_state

        G = 0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            returns[state].append(G)
            value_table[state] = np.mean(returns[state])

    return value_table

fv_value_function = first_visit_mc(env, policy)
ev_value_function = every_visit_mc(env, policy)

# 일부 결과 출력
print("First-Visit MC 결과:")
for state, value in list(fv_value_function.items())[:5]:
    print(f"State: {state}, Value: {value:.4f}")

print("\nEvery-Visit MC 결과:")
for state, value in list(ev_value_function.items())[:5]:
    print(f"State: {state}, Value: {value:.4f}")