## Snake

I wanted an environment to experiment with reinforcement learning in the game Snake, so I made one. A few refactors later and the game is made from three easily resuable components (game, agent, interface). Hooray for Python. Trying to follow patterns established by openai gym et al., to play the game we need an *agent* who, for every frame, fetches a *game state*, makes a decision, and finally takes a *step* with an *action* to get a *reward* and advance to the next frame. Additionally the agent may use an *interface* for display and user input.

![screencap](https://github.com/tyoungNIO/snake-python/blob/master/screencap.png)

To play the game using the arrow keys on your keyboard, run `python agent_human.py`.

To get a benchmark of random performance for your machine learning needs, run `python agent_random.py`.

To develop an agent, follow this example and/or the included agents. In addition to eating apples, an agent should note which direction is backwards and avoid moving that way to prevent an immediate death.
```
from human_interface import Interface
from snake import Snake


game = Snake()
interface = Interface(
    grid_size=game.feature_space(),
    dot_size=32,
    actions=game.actions(),
    fps=10)

game_over = False
while not game_over:
    apple, snake, score = game.game_state()
    interface.draw_frame(apple, snake, score)
    # select an action from game.actions(), or
    # action = interface.get_user_input()
    reward, game_over = game.step(action)
```
