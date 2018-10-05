## Snake

I wanted an environment to experiment with reinforcement learning in the game Snake, so I made one. A few refactors later and the game is made from three easily resuable components (game, agent, interface). Hooray for Python. Trying to follow patterns established by openai gym et al., to play the game we need an *agent* who, for every frame, fetches a *game state*, makes a decision, and finally takes a *step* with an *action* to advance to the next frame. Additionally the agent may use an *interface* for display and user input.

To play the game using the arrow keys on your keyboard, run `python agent_human.py`. To get a benchmark of random performance for your machine learning needs, run `python agent_random.py`.
