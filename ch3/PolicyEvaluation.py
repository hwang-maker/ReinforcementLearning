from Environments.GridWorld import GridworldEnv

env = GridworldEnv()
print("총 상태 개수:", env.observation_space.n)  # 16
print("총 행동 개수:", env.action_space.n)  # 4

env.reset()
env.render()

V = env.evaluate_policy()
print("정책 평가 결과 (상태 가치 함수):")
print(V.reshape(env.grid_size))

