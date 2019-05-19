import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')
tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = 0.99  # discounted future reward
e = 0.1  # chance of random move
num_episodes = 1000
# create lists to contain total rewards and steps per episode
winners = []
losers =  []
rewards = []
with tf.Session() as sess:
    sess.run(init)
    for r in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        total_rewards = 0
        done = False
        step = 0
        # The Q-Network
        while step < 99:
            step += 1
            # Choose an action with e chance of random action to help
            # early training
            action, allQ = sess.run(
                [predict, Qout],
                feed_dict = {inputs1: np.identity(16)[state: state + 1]})
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()
            # Get new state and reward from environment
            observation, reward, done, info = env.step(action[0])
            # observation is the index of player position on grid

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(
                Qout,
                feed_dict={
                    inputs1: np.identity(16)[observation: observation + 1]
                })
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, action[0]] = reward + y * maxQ1
            # Train our network using target and predicted Q values
            _, W1 = sess.run(
                [updateModel, W],
                feed_dict={
                    inputs1: np.identity(16)[state: state + 1],
                    nextQ: targetQ
                })
            total_rewards += reward
            state = observation
            if done:
                if reward:
                    winners.append(step)  # steps to success
                    losers.append(-1)
                else:
                    winners.append(-1)
                    losers.append(step)  # steps to failure
                e = 1 / ((r / 50) + 10)  # Reduce chance of random action as we train the model.
                break
            if step == 99:
                winners.append(-1)
                break
        rewards.append(total_rewards)

print("Total Success Rate: {}".format(sum(rewards) / num_episodes))
plt.plot(winners, 'g.')  # should trend down
plt.plot(losers, 'r+')  # should trend up
plt.axis([0, num_episodes, 0, 99])
plt.show()
