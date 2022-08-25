from main import Agent

from game import Snake_Game_Agent

import time


game = Snake_Game_Agent()

agent = Agent()
agent.n_games = 40
i = 1
total_score = 0
while True:
    try:
        action = agent.get_action(agent.get_state(game))
        reward,alive,score = game.play(action)
        if not alive:
            time.sleep(0.4)
            game.reset()
            print(f"Game : {i}, Score : {score}")
            i += 1
            total_score += score
    except KeyboardInterrupt:
        print(f"Mean Score : {total_score/(i-1)} for {i-1} games.")
        break