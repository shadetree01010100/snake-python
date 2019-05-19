import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from human_interface import Interface
from snake import Snake


class Agent():

    def __init__(self, game, dot_size=32, trials=1000, learning_rate=0.001):
        self.game = game
        self.actions = self.game.actions()
        self.interface = Interface(
            self.game.feature_space(), dot_size, self.actions)
        self.trials = trials
        # results will go in these lists
        self.frames = []
        self.scores = []
        self.accumulated_rewards = []
        self.random_choice_history = []
        # build network
        self._setup_qnn(learning_rate)
        # learning params
        self.discounted_future_reward = 0.1
        self.max_random_action_prob = 1 / 20
        self.random_action_prob = self.max_random_action_prob
        self.random_action_decay = self.trials / 100
        # limit the steps of lost agents
        self.max_idle_frames = self.game.feature_space()[0] * \
                               self.game.feature_space()[1] / 2

    def run(self):
        """ Learn all the things."""
        action_list = list(self.game.actions().values())
        i = 0
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(
                './train/{}'.format(int(time.time())),
                sess.graph,
            )
            sess.run(tf.global_variables_initializer())
            for r in range(self.trials):
                self.random_choice_history.append(
                        self.random_action_prob * 100)
                # first observation
                apple, snake, score = self.game.game_state()
                accumulated_reward = 0
                done = False
                step = 0
                # The Q-Network
                while step < self.max_idle_frames:
                    inputs = self.get_inputs(apple, snake, score)
                    prediction, allQ = sess.run(
                        [self.predict, self.Qout],
                        feed_dict = {
                            self.inputs: [inputs],
                        },
                    )
                    if np.random.rand(1) < self.random_action_prob:
                        action_index = random.randint(0, 3)
                    else:
                        action_index = prediction[0]
                    action = list(self.actions.values())[action_index]
                    reward, game_over = self.game.step(action)
                    new_apple, new_snake, new_score = self.game.game_state()
                    # was that a good choice of action for that state?
                    inputs = self.get_inputs(new_apple, new_snake, new_score)
                    Q1 = sess.run(
                        self.Qout,
                        feed_dict={
                            self.inputs: [inputs],
                        },
                    )
                    # set target for action
                    maxQ1 = np.max(Q1)
                    targetQ = allQ
                    last_delta_apple = self.get_inputs(apple, snake, score)[:2]
                    this_delta_apple = self.get_inputs(
                        new_apple, new_snake, new_score)[:2]
                    x_reward = last_delta_apple[0] - this_delta_apple[0]
                    x_reward *= self.discounted_future_reward
                    y_reward = last_delta_apple[1] - this_delta_apple[1]
                    y_reward *= self.discounted_future_reward
                    mini_reward = max([x_reward, y_reward, reward])
                    # if not mini_reward:
                        # # penalty?
                        # mini_reward = self.discounted_future_reward * -1
                    if reward > 0:
                        actual_reward = reward * maxQ1
                    elif reward < 0:
                        # actual_reward = reward
                        actual_reward = 0  # prevent negative loss
                    else:
                        actual_reward = mini_reward * maxQ1
                    # actual_reward = reward * maxQ1
                    accumulated_reward += actual_reward
                    targetQ[0, prediction[0]] = actual_reward
                    # train
                    inputs = self.get_inputs(apple, snake, score)
                    summary, _, loss, W0 = sess.run(
                        [
                            self.summaries,
                            self.train_func,
                            self.loss_func,
                            self.weights_0,
                        ],
                        feed_dict={
                            self.inputs: [inputs],
                            self.nextQ: targetQ,
                        }
                    )
                    if i % 1000 == 0:
                        summary_writer.add_summary(summary, i)
                    # self.interface.draw_frame(
                        #new_apple, new_snake, new_score)
                    self.interface.draw_frame(
                        new_apple, new_snake, r)
                    self.interface.pump()  # clear event queue
                    if reward:
                        # reset counter to max_idle_frames
                        step = 0
                    if game_over or step > self.max_idle_frames:
                        break
                    apple, snake, score = new_apple, new_snake, new_score
                    step += 1
                    i += 1
                # # draw final frame:
                # self.interface.draw_frame(apple1, snake1, score1)
                # self.interface.pump()  # clear event queue
                self._decay_random_chance(r)
                self.frames.append(step)
                self.scores.append(score)
                self.accumulated_rewards.append(accumulated_reward)
                self.game.reset()
            summary_writer.close()
            self.interface.close(True)

    def get_action(self):
        return random.choice(list(self.actions.values()))

    def get_inputs(self, apple, snake, score):
        """ Returns 10 values representing the snake's position:
          - Delta X and Y to apple
          - Distance from head to all 4 walls to the left, right, up, down
          - Distance from head to the nearest body segment in each direction
        """
        head_x = snake[0][0]
        head_y = snake[0][1]
        # delta to apple is easy
        apple_dx = apple[0] - head_x
        apple_dy = apple[1] - head_y
        # walls are also easy
        max_x, max_y = self.game.feature_space()
        wall_l = head_x + 1
        wall_r = max_x - head_x
        wall_u = head_y + 1
        wall_d = max_y - head_y
        if not score:  # the snake has no body, we're done here
            body_l = 0
            body_r = 0
            body_u = 0
            body_d = 0
            return apple_dx, apple_dy, \
                   wall_l, wall_r, wall_u, wall_d, \
                   body_l, body_r, body_u, body_d
        # distance from head to nearest body segment in all 4 directions...
        # list all body segments on the same x or y coord as the head:
        x_segments = [s for s in snake[1:] if s[0] == head_x]
        y_segments = [s for s in snake[1:] if s[1] == head_y]
        # list all segments with common y and lesser x,
        # i.e. vertically aligned with and left of the head
        l_segments = [y[0] for y in y_segments if y[0] < head_x]
        # vertically aligned with and right of the head
        r_segments = [y[0] for y in y_segments if y[0] > head_x]
        # horizontally aligned with head, above and below
        u_segments = [x[1] for x in x_segments if x[1] < head_y]
        d_segments = [x[1] for x in x_segments if x[1] > head_y]
        # distance to nearest body segment to the left
        try:
            body_l = head_x - max(l_segments)
        except ValueError:  # the the segment list is empty
            body_l = 0
        # nearest to the right
        try:
            body_r = min(r_segments) - head_x
        except ValueError:
            body_r = 0
        # above and below
        try:
            body_u = head_y - max(u_segments)
        except ValueError:
            body_u = 0
        try:
            body_d = min(d_segments) - head_y
        except ValueError:
            body_d = 0

        return apple_dx, apple_dy, \
               wall_l, wall_r, wall_u, wall_d, \
               body_l, body_r, body_u, body_d

    def _decay_random_chance(self, r):
        # reduce random action chance
        # denom = (r / self.random_action_decay) + self.max_random_action_prob
        # self.random_action_prob = 1 / denom
        # print(self.random_action_prob)
        pass

    def _setup_qnn(self, learning_rate):
        # feed-forward QNN
        # 10 inputs from self.get_inputs()
        inputs = 10
        # hidden_0 = 16
        outputs = len(self.game.actions())  # 4
        output_bias = tf.Variable(tf.ones([outputs]))
        self.inputs = tf.placeholder(shape=[1, inputs], dtype=tf.float32)
        self.weights_0 = tf.Variable(
            tf.random_uniform([inputs, outputs], -1.0, 1.0))
        # self.weights_1 = tf.Variable(
            # tf.random_uniform([hidden_0, outputs], -2.0, 2.0))
        # self.weights_0 = tf.Variable(tf.ones([inputs, outputs]))
        # self.hidden_0 = tf.matmul(self.inputs, self.weights_0)
        self.Qout_logits = tf.matmul(self.inputs, self.weights_0) + output_bias
        self.Qout = tf.nn.softmax(self.Qout_logits)
        self.predict = tf.argmax(self.Qout, axis=1)
        # loss function
        self.nextQ = tf.placeholder(shape=[1, outputs], dtype=tf.float32)
        self.loss_func = tf.losses.softmax_cross_entropy(
            self.nextQ, self.Qout_logits)
        # self.loss_func = tf.losses.absolute_difference(self.nextQ, self.Qout)
        # self.loss_func = tf.reduce_sum(tf.square(self.nextQ - self.Qout))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_func = optimizer.minimize(self.loss_func)
        # tensorboard
        tf.summary.scalar('loss', self.loss_func)
        tf.summary.histogram('weights_0', self.weights_0)
        # tf.summary.histogram('weights_1', self.weights_1)
        self.summaries = tf.summary.merge_all()

if __name__ == '__main__':
    game = Snake(
        # grid_size=(16, 16),
        # random_spawn=True,
        # seed=8675309,
    )
    agent = Agent(
        game=game,
        # dot_size=32,
        trials=10000,
        learning_rate=0.00001,
    )
    agent.run()
    plt.plot(agent.scores)
    plt.plot(agent.accumulated_rewards)
    # plt.plot(agent.random_choice_history)
    # plt.axis([0, agent.trials, -5, 5])
    plt.show()
