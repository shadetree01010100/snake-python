import time
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from human_interface import Interface
from snake import Snake


# suppress tf build info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        self.max_random_action_prob = 0.3
        self.random_action_prob = self.max_random_action_prob
        self.random_action_cutoff = self.trials / 2
        self.discounted_future_reward = 0.9
        # self.max_dfr_prob = 0.5
        # self.dfr_prob = self.max_dfr_prob
        # self.dfr_cutoff = self.trials / 2
        # limit the steps of lost agents
        self.max_idle_frames = (self.game.feature_space()[0] *
                                self.game.feature_space()[1])# // 2

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
                    start_time = time.time()
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
                    apple_l, apple_r, apple_d, apple_u = self.get_inputs(
                        apple, snake, score)[:4]
                    new_apple_l, new_apple_r, new_apple_d, new_apple_u = self.get_inputs(
                        new_apple, new_snake, new_score)[:4]
                    x_reward = 0
                    y_reward = 0
                    if (new_apple_l < apple_l) or (new_apple_r < apple_r):
                        x_reward = self.discounted_future_reward
                    if (new_apple_d < apple_d) or (new_apple_u < apple_u):
                        y_reward = self.discounted_future_reward
                    mini_reward = x_reward or y_reward
                    # if not mini_reward:
                    #     # wrong way, dummy. penalty???
                    #     mini_reward = self.discounted_future_reward * -1
                    if reward > 0:
                        # success
                        mini_reward = 0
                        actual_reward = reward
                        # all other choices would have been wrong
                        # targetQ[0] = [0 for _ in targetQ[0]]
                        targetQ[0, prediction[0]] = actual_reward
                    elif reward == 0:
                        # apply mini reward if we got closer
                        actual_reward = max(mini_reward, 0)
                        targetQ[0, prediction[0]] = actual_reward
                    else:
                        # death
                        # literally anything else would have been better
                        actual_reward = reward
                        targetQ[0] = [1 for _ in targetQ[0]]
                        # don't do that again
                        targetQ[0, prediction[0]] = 0
                    accumulated_reward += actual_reward
                    # train
                    inputs = self.get_inputs(apple, snake, score)
                    # summary, _, loss, W0, W1 = sess.run(
                    summary, _, loss, W0 = sess.run(
                        [
                            self.summaries,
                            self.train_func,
                            self.loss_func,
                            # self.weights_0,
                            # self.weights_1,
                            self.weights,
                        ],
                        feed_dict={
                            self.inputs: [inputs],
                            self.nextQ: targetQ,
                        }
                    )
                    if i % 1000 == 0:
                        summary_writer.add_summary(summary, i)
                    if reward:
                        # reset counter to max_idle_frames
                        step = 0
                    if game_over or step > self.max_idle_frames:
                        break
                    apple, snake, score = new_apple, new_snake, new_score
                    step += 1
                    i += 1
                    self.interface.draw_frame(new_apple, new_snake, r)
                    self.interface.pump()  # clear event queue
                    end_time = time.time()
                    time_delta = end_time - start_time
                    try:
                        time.sleep(1/60 - time_delta)
                    except ValueError:
                        pass
                # self.interface.draw_frame(new_apple, new_snake, r)
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
        apple_l = 0
        apple_r = 0
        apple_d = 0
        apple_u = 0
        if apple[0] < head_x:
            # to the left
            apple_l = abs(head_x - apple[0])
            apple_r = 0
        elif apple[0] > head_x:
            # to the right
            apple_l = 0
            apple_r = abs(head_x - apple[0])
        if apple[1] < head_y:
            # above
            apple_d = 0
            apple_u = abs(head_y - apple[1])
        elif apple[1] > head_y:
            # below
            apple_d = abs(head_y - apple[1])
            apple_u = 0
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
            return apple_l, apple_r, apple_u, apple_d, \
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

        return apple_l, apple_r, apple_u, apple_d, \
               wall_l, wall_r, wall_u, wall_d, \
               body_l, body_r, body_u, body_d

    def _decay_random_chance(self, r):
        # reduce random action chance
        percent = max(1 - (r / self.random_action_cutoff), 0)
        self.random_action_prob = self.max_random_action_prob * percent

    def _setup_qnn(self, learning_rate):
        # feed-forward QNN
        input_nodes = 12  # 12 inputs from self.get_inputs()
        # hidden_nodes = 8
        output_nodes = len(self.game.actions())  # 4 directions
        self.inputs = tf.placeholder(shape=[1, input_nodes], dtype=tf.float32)
        # self.weights_0 = tf.Variable(tf.ones([input_nodes, hidden_nodes]))
        self.weights = tf.Variable(tf.ones([input_nodes, output_nodes]))
        # hidden_logits = tf.matmul(self.inputs, self.weights_0)
        # self.hidden = tf.sigmoid(hidden_logits)
        # self.weights_1 = tf.Variable(tf.ones([hidden_nodes, output_nodes]))
        # self.Qout_logits = tf.matmul(self.hidden, self.weights_1)
        self.Qout_logits = tf.matmul(self.inputs, self.weights)
        self.Qout = tf.nn.sigmoid(self.Qout_logits)
        self.predict = tf.argmax(self.Qout, axis=1)
        # loss function
        self.nextQ = tf.placeholder(shape=[1, output_nodes], dtype=tf.float32)
        self.loss_func = tf.losses.sigmoid_cross_entropy(
            self.nextQ, self.Qout_logits)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_func = optimizer.minimize(self.loss_func)
        # tensorboard
        tf.summary.scalar('loss', self.loss_func)
        # tf.summary.histogram('weights_0', self.weights_0)
        # tf.summary.histogram('weights_1', self.weights_1)
        tf.summary.histogram('weights', self.weights)
        self.summaries = tf.summary.merge_all()

if __name__ == '__main__':
    game = Snake(
        grid_size=(8, 8),
        random_spawn=True,
        # seed='SeymourButts',
    )
    agent = Agent(
        game=game,
        # dot_size=32,
        trials=10000,
        learning_rate=0.001,
    )
    try:
        agent.run()
    except KeyboardInterrupt:
        pass
    print('HIGH SCORE: {}'.format(max(agent.scores)))
    plt.plot(agent.scores)
    # plt.plot(agent.accumulated_rewards)
    plt.plot(agent.random_choice_history)
    # plt.axis([0, agent.trials, -5, 5])
    plt.show()
