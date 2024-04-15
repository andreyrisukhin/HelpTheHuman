from pettingzoo.test import parallel_api_test
import os

from src.color_maze import ColorMaze
from src.manual_policy import ManualPolicy


'''
Things to debug:
> Follower and Leader have two moves to each side, change print for the user
> Grid is 6x11
> Make sure leader and follower are spawned first, then blocks. 
'''

if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    env = ColorMaze()
    parallel_api_test(env)
    observations, _ = env.reset(seed=42)
    manual_policy_1 = ManualPolicy(env, agent_id=0)
    manual_policy_2 = ManualPolicy(env, agent_id=1)
    env.render()
    score = 0

    while env.agents:
        actions = {
            'leader': manual_policy_1(observations[env.agents[0]], agent=env.agents[0]),
            'follower': manual_policy_2(observations[env.agents[1]], agent=env.agents[1])
        }
        observations, rewards, _, _, _ = env.step(actions)
        score += rewards['leader']

        os.system('cls' if os.name == 'nt' else 'clear')
        env.render()
        print(f'Score: {score} | Goal: {env.goal_block}')

    env.close()
