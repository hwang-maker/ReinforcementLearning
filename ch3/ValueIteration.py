import numpy as np
from Environments.GridWorld import GridworldEnv

env = GridworldEnv()

def value_iteration(env, theta=1e-6, discount_factor =1.0):

    def one_step_lookahead(state,V):
        A = np.zeros(env.action_space.n)

        for a in range(env.action_space.n):
            for next_state in range(env.observation_space.n):
                prob = env.P[a,state,next_state] # numpy 배열 에서는 [a,state,next_state]와 같이 call 해야함! [a][state][next_state] (x)
                A[a] += prob * (env.R[a,state] + discount_factor * V[next_state])

        return A

    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0

        for s in range(env.observation_space.n):
            A = one_step_lookahead(s,V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))

            V[s] = best_action_value

        if delta < theta:
            break

    policy = np.zeros([env.observation_space.n, env.action_space.n])

    for s in range(env.observation_space.n):
        A = one_step_lookahead(s,V)
        best_action = np.argmax(A)

        policy[s, best_action] = 1.0

    return policy, V



policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.grid_size))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.grid_size))
print("")

