from Environments.GridWorld import GridworldEnv
import numpy as np

def policy_iteration(env, max_iterations=1000):
    """
    Policy Iteration Algorithm:
    1. Evaluate the policy
    2. Improve the policy
    3. Repeat until the policy is stable

    Args:
        env: GridworldEnv
        max_iterations: 최대 반복 횟수

    Returns:
        optimal_policy: 최적 정책
        optimal_V: 최적 가치 함수
    """
    policy_stable = False
    iteration = 0

    while not policy_stable and iteration < max_iterations:
        iteration += 1
        print(f"🚀 Iteration {iteration}: Policy Evaluation & Improvement")

        # Step 1: 정책 평가 (Policy Evaluation)
        V = env.evaluate_policy()

        # Step 2: 정책 개선 (Policy Improvement)
        new_policy, new_V = env.policy_improvement()

        # 정책이 변하지 않으면 종료
        if np.array_equal(env.policy, new_policy):
            policy_stable = True

        env.policy = new_policy  # 새로운 정책 적용
        print(f"✅ Policy Iteration {iteration} Completed!")

    return env.policy, V



# 실행 코드
env = GridworldEnv()  # Gridworld 환경 생성
optimal_policy, optimal_V = policy_iteration(env)

# 최적 정책 및 가치 함수 출력
print("\n🔹 Policy probability Distribution")
print(optimal_policy)

print("\n Reshaped Grid Policy(0=up, 1=right, 2=down, 3=left)")
print(np.reshape(np.argmax(optimal_policy,axis=1), env.grid_size))

print("\n🔹 최적 가치 함수:")
print(optimal_V.reshape(env.grid_size))
