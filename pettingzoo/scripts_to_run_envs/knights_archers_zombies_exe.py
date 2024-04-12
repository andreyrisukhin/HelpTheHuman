import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10

env = knights_archers_zombies_v10.env(render_mode="human")
env.reset(seed=42)

manual_policy = knights_archers_zombies_v10.ManualPolicy(env)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    elif agent == manual_policy.agent:
        # get user input (controls are WASD and space)
        action = manual_policy(observation, agent)
    else:
        # this is where you would insert your policy (for non-player agents)
        action = env.action_space(agent).sample()

    env.step(action)
env.close()