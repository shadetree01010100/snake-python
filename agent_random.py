import random
from human_interface import Interface
from snake import Snake


class Agent():

    def __init__(self, game, dot_size=32, trials=100):
        self.game = game
        self.actions = self.game.actions()
        self.interface = Interface(
            self.game.feature_space(), dot_size, 0, self.actions)
        self.trials = trials
        # results will go in these lists
        self.frames = []
        self.scores = []

    def run(self):
        """ Benchmark performance over `trials` making a random move each
        frame. Displays the last frame of each trial and prints the average
        frames survived and average score for all agents.
        """
        for r in range(self.trials):
            game_over = False
            frame_num = 0
            action = random.choice(list(self.actions.values()))
            # the snake can't go backwards, define that action
            backwards = (action[0] * -1, action[1] * -1)
            while not game_over:
                frame_num += 1
                reward, game_over = self.game.step(action)
                apple, snake, score = self.game.game_state()
                if game_over:
                    break
                action = random.choice(list(self.actions.values()))
                while action == backwards:
                    action = random.choice(list(self.actions.values()))
                backwards = (action[0] * -1, action[1] * -1)
            self.frames.append(frame_num)
            self.scores.append(score)
            self.interface.draw_frame(apple, snake, score)
            self.game.reset()
        print('average frames: {}'.format(sum(self.frames) / self.trials))
        print('average score: {}'.format(sum(self.scores) / self.trials))
        self.interface.close(True)

if __name__ == '__main__':
    game = Snake(
        # grid_size=(16, 16),
        # random_spawn=False,
        # seed=None,
    )
    agent = Agent(
        game=game,
        # dot_size=32,
        # trials=100,
    )
    agent.run()
