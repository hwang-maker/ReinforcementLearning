from GridWorld import GridWorld
from QAgent import QAgent

def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000): # 1000회 동안 학습
        done = False
        history = []

        s = env.reset()
        while not done: #한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        agent.update_table(history)  # 히스토리를 이용하여 에이전트를 업데이트
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    print("0번: 왼쪽 1번: 위, 2번: 오른쪽 3번: 아래쪽")
    main()