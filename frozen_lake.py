"""
Q-Table algorithm implemented in the FrozenLake environment
https://gym.openai.com/envs/FrozenLake-v0/
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
"""

import os
import gym
import matplotlib.pyplot as plt
import numpy as np


env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
learning_rate = .8
y = .95  # discounted future reward
num_episodes = 1000  # imporovement plateaus around 1k
# create lists to contain total rewards and steps per episode
winners = []
losers = []
rewards = []
for r in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    if r == 0:
        os.system('cls')
        env.render()
    total_rewards = 0
    done = False
    step = 0
    #The Q-Table learning algorithm
    while step < 99:
        step += 1
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1,env.action_space.n) * (1. / (r + 1)))
        # Get new state and reward from environment
        observation, reward, done, info = env.step(action)
        # Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + learning_rate*(reward + y * np.max(Q[observation, :]) - Q[state, action])
        total_rewards += reward
        state = observation
        if done:
            if reward:
                winners.append(step)  # steps to success
                losers.append(-1)
            else:
                winners.append(-1)
                losers.append(step)  # steps to failure
            break
        if step == 99:
            winners.append(-1)
            break
    rewards.append(total_rewards)

print("Total Success Rate: {}".format(sum(rewards) / num_episodes))
# print("Final Q-Table Values:")
# print(Q)
plt.plot(winners, 'g.') # should trend down
plt.plot(losers, 'r+')  # should trend up
plt.axis([0, num_episodes, 0, 99])
plt.show()
