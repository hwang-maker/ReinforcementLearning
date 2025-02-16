from GridWorld import GridWorld
from Agent import Agent


def main():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]
    gamma = 1.0
    alpha = 0.01

    for k in range(50000):
        done = False
        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            (x_prime, y_prime), reward, done = env.step(action)
            x_prime, y_prime = env.get_state()

            data[x][y] = data[x][y] + alpha*(reward + gamma * data[x_prime][y_prime] - data[x][y])

        env.reset()

    for row in data:
        print(row)


if __name__ == '__main__':
    main()