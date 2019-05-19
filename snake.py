import random


class Snake():

    def __init__(self, grid_size=(16, 16), random_spawn=False, seed=None):
        """ Game ends if the snake's head touches its body or goes out of
        bounds. Eat apples to get a reward and grow the snakes body by one
        segment. Agents should call `actions()` and `feature_space()` to get
        valid actions and boundaries, and then `step()` with an `action` to
        advance one frame in the game.

        Args:
          - grid_size: (tuple) play area dimensions x, y
          - random_spawn: (bool) if False, player will begin centered
          - seed: if not None, randomly generated numbers will be repeatable
        """
        self.width, self.height = grid_size
        self.random_spawn = random_spawn
        self.seed = seed
        self.reset()

    def actions(self):
        """ Returns a map of valid actions as `{key: action}` pairs."""
        return {
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
            'UP': (0, -1),
            'DOWN': (0, 1),
        }

    def feature_space(self):
        """ Returns the play area dimensions."""
        return (self.width, self.height)

    def game_state(self):
        """ Returns the game state."""
        return self.apple, self.snake, self.score

    def reset(self):
        """ Call to reset the game. If a random seed has been set random
        numbers, such as the sequence of apples, will be repeated exactly.
        """
        # the snake is simply a list of coordinates, with length 1 to start
        if self.seed != None:
            random.seed(self.seed)
        if self.random_spawn:
            self.snake = [self._random_coords()]
        else:
            center = (self.width // 2, self.height // 2)
            self.snake = [center]
        # don't spawn apple on top of snake
        self.apple = self._random_coords(self.snake)
        self.game_over = False
        self.score = 0

    def step(self, action):
        """ Advance one frame in the game.

        Args:
          - action: (tuple) direction to move, must be a value from
            `actions()` otherwise an `ValueError` is raised
        Returns:
          - apple: (tuple) coordinates of the apple
          - snake: (list) tuples of coordinate pairs of each segment of the
            snake's body, ordered from head to tail
          - reward: (int) 1 if an apple is consumed, -1 if the player dies,
            otherwise 0
          - game_over: (bool) if True further calls to this method will raise
            an `Exception` until `reset()` has been called
        """
        if self.game_over:
            raise Exception('Game Over'.format(self.score))
        if action not in self.actions().values():
            raise ValueError('Invalid action \"{}\"'.format(action))
        reward = 0
        # get new head position from action vector
        current_head = self.snake[0]
        new_head = (
            current_head[0] + action[0],
            current_head[1] + action[1],
        )
        # grow if found apple
        if new_head == self.apple:
            self.score += 1
            reward = 1
            # don't spawn new apple on top of snake
            excluded = [new_head] + self.snake[:self.score]
            self.apple = self._random_coords(excluded)
        # move the snake by prepending a new head and truncating the tail
        # because we move the snake before checking for collisons we also
        # need to check for special case of reversing with 1 body segment
        if self.score == 1:
            if new_head == self.snake[-1]:
                self.game_over = True
                reward = -1
        self.snake = [new_head] + self.snake[:self.score]
        # check for collisions
        if new_head in self.snake[1:] or \
                new_head[0] not in range(self.width) or \
                new_head[1] not in range(self.height):
            self.game_over = True
            reward = -1
        return reward, self.game_over

    def _random_coords(self, excluded=[]):
        """ Create a random coordinate pair. Does not return until a random
        coordinate is found that is not in `excluded`, potentially forever if
        a valid value cannot be found.

        Args:
          - excluded: a list of coordinate tuples that will not be returned
        Returns:
          - coords: (tuple) x, y integers
        """
        def new_coords():
            return (
                random.randint(0, self.width - 1),
                random.randint(0, self.height - 1),
            )
        coords = new_coords()
        while coords in excluded:
            coords = new_coords()
        return coords
 