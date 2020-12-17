import time
import pygame
from colors import *

class Interface():

    break_keys = [pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE]

    def __init__(self, grid_size, dot_size, actions, fps=0):
        """ Provides methods to get user input and draw the game to screen.

        Args:
          - grid_size: (tuple) x, y dimensions of play area
          - dot_size: (int) diameter of points when drawing frame, must be
            greater than 1, and probably should be even
          - actions: (dict) map of {key: action} pairs
          - fps: (int) pause for `1 / fps` seconds before returning from
            `draw_frame`, limiting max frames per second to this value
        """
        self.grid_size = grid_size
        self.dot_size = dot_size
        self.fps = fps
        self.action_map = {}
        for label, action in actions.items():
            key = getattr(pygame, 'K_{}'.format(label))
            self.action_map[key] = action
        pygame.init()
        self.window_size = (
            self.grid_size[0] * self.dot_size + self.dot_size,
            self.grid_size[1] * self.dot_size + self.dot_size,
        )
        self.surface = pygame.display.set_mode(self.window_size)

    def close(self, done=False):
        """ Shutdown the game, optionally holding open the game window.

        Args:
          - done: (bool) if True, close the game window and exit immediately,
            otherwise blocks and waits for one of `break_keys`.
        """
        while done == False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # ctrl+C or closed window
                    done = True
                    break
                if event.type == pygame.KEYDOWN:
                    ctrl_key = pygame.key.get_mods() & pygame.KMOD_CTRL
                    if ctrl_key and event.key == pygame.K_c:
                        done = True
                        break
                    if event.key in self.break_keys:
                        done = True
                        break
        pygame.display.quit()

    def draw_frame(self, apple, snake, title):
        """ Draw the current game state.

        Args:
          - apple: (tuple) x, y coordinates of apple
          - snake: (list) sequence of coordinates representing snake
          - score: (int) displayed in window title
        """
        self.surface.fill(BLACK)
        self._draw_walls()
        self._draw_apple(apple)
        self._draw_snake(snake)
        pygame.display.set_caption(title)
        pygame.display.update()
        if self.fps > 0:
            time.sleep(1 / self.fps)

    def get_user_input(self, excluded=[]):
        """ Get keystrokes from user. Returns the `action` mapped to the `key` pressed, or `None` if no valid input is taken.

        Args:
          - excluded: (list) sequence of values in `self.action_map` that
            will be ignored
        """
        for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key not in self.action_map.keys():
                        continue  # not a mapped key, ignore it
                    new_action = self.action_map[event.key]
                    if new_action in excluded:
                        continue  # ignore repeated or backwards moves
                    action = new_action
                    return action  # done taking user input

    def pump(self):
        """ experimental"""
        pygame.event.pump()

    def _draw_apple(self, apple):
        pygame.draw.circle(
            self.surface,
            RED,
            self._scale_coords(apple),
            self.dot_size // 2, # radius
        )

    def _draw_snake(self, snake):
        for i, unit in enumerate(snake):
            alpha = 1 - (i / len(snake))
            if unit == snake[0]:  # check for collision with head
                alpha = 1.0  # redraw the head at this coordinate
            color = blend_colors(WHITE, GREEN, alpha)
            pygame.draw.circle(
                self.surface,
                color,
                self._scale_coords(unit),
                self.dot_size // 2,  # radius
            )

    def _draw_walls(self):
        width, height = self.window_size
        offset = self.dot_size // 2
        pygame.draw.line(
            self.surface,
            RED,
            (offset, offset),
            (width - offset, offset),
        )
        pygame.draw.line(
            self.surface,
            RED,
            (width - offset, offset),
            (width - offset, height - offset),
        )
        pygame.draw.line(
            self.surface,
            RED,
            (width - offset, height - offset),
            (offset, height - offset),
        )
        pygame.draw.line(
            self.surface,
            RED,
            (offset, height - offset),
            (offset, offset),
        )

    def _scale_coords(self, coords):
        """ Scale up points by `dot_size`, and offset objects by half that
        value from the edges.
        """
        return (
            coords[0] * self.dot_size + self.dot_size,
            coords[1] * self.dot_size + self.dot_size,
        )
