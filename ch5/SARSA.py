from ch5.GridWorld import GridWorld
from ch5.QAgent import SAgent

def main():
    env = GridWorld()
    agent = SAgent()

    for n_epi in range(1000):
        done = False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s = s_prime
        agent.anneal_eps()
    agent.show_table()

if __name__ == '__main__':
    print("0번: 왼쪽 1번: 위, 2번: 오른쪽 3번: 아래쪽")
    main()