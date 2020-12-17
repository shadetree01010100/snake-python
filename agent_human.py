import random
from human_interface import Interface
from snake import Snake


class Agent():

    def __init__(self, game, dot_size=32, fps=10):
        self.game = game
        self.actions = self.game.actions()
        self.interface = Interface(
            self.game.feature_space(), dot_size, self.actions, fps)

    def run(self):
        """ Play the game."""
        action = random.choice(list(self.actions.values()))
        # the player can't go backwards, define that action
        backwards = (action[0] * -1, action[1] * -1)
        game_over = False
        while not game_over:
            # advance one frame
            reward, game_over = self.game.step(action)
            apple, snake, score = self.game.game_state()
            self.interface.draw_frame(apple, snake, 'Score: {}'.format(score))
            if game_over:
                break
            new_action = self.interface.get_user_input([action, backwards])
            if new_action is not None:
                action = new_action
                backwards = (action[0] * -1, action[1] * -1)
        print('GAME OVER: {}'.format(score))
        self.interface.close()

if __name__ == '__main__':
    game = Snake(
        # grid_size=(16, 16),
        # random_spawn=False,
        # seed=None,
    )
    agent = Agent(
        game=game,
        # dot_size=32,
        # fps=10,
    )
    agent.run()
