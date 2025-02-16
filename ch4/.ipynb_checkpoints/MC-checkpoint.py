from GridWorld import GridWorld
from Agent import Agent

def main():

    env = GridWorld()
    agent = Agent()
    data = [[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0]] # table initailization
    gamma = 1.0
    alpha = 0.0001

    for k in range(50000):
        done = False
        history = []
        while not done:
            action = agent.select_action()
            (x,y), reward, done = env.step(action)
            history.append((x,y,reward))
        env.reset()

        cum_reward = 0 # Return
        for transition in history[::-1]: # 리스트를 거꾸로 순회
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            cum_reward = reward + gamma*cum_reward

        i = 0
        for row in data:
            if i % 4 == 0:
                print("[",end="")
                print(row)

            elif i % 3 == 0:
                print(row,end="")
                print("]")
            else:
                print(row)
            i+=1

if __name__ == '__main__':
    main()