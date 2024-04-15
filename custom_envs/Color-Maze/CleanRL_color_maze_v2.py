import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Any, Mapping

from src import color_maze


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.critic = nn.Linear(64, 1)
        self.actor = nn.Linear(64, action_space.n)

    def forward(self, x):
        features = self.network(x)
        return self.actor(features), self.critic(features)


def collect_data(
        env: color_maze.ColorMaze,
        models: Mapping[str, nn.Module],
        num_steps: int
):
    obs, _ = env.reset()
    data = {agent: [] for agent in env.agents}
    for _ in range(num_steps):
        action_log_probs = {}
        values = {}
        actions = {}
        for agent in env.agents:
            model = models[agent]
            # Unsqueeze observation to have batch size 1 and flatten the grid into 1-dimension
            obs_tensor = torch.tensor(obs[agent], dtype=torch.float32).unsqueeze(0).flatten(start_dim=1, end_dim=-1)
            with torch.no_grad():
                logits, value = model(obs_tensor)
                # TODO figure out why logits are sometimes NaN which causes an error here
                dist = Categorical(logits=logits)
                action = dist.sample()

            actions[agent] = action.item()
            action_log_probs[agent] = dist.log_prob(action)
            values[agent] = value.item()

        obs, rewards, terminateds, truncations, _ = env.step(actions)

        for agent in env.agents:
            data[agent].append((obs_tensor, action.item(), rewards[agent], terminateds[agent], action_log_probs[agent], values[agent]))

        if terminateds[env.agents[0]]:
            # The environment terminates for all agents at the same time
            break

    return data


def ppo_update(
        models: Mapping[str, nn.Module],
        optimizers: Mapping[str, optim.Optimizer],
        data: dict[str, Any],
        epochs: int,
        gamma: float,
        clip_param: float
):
    acc_losses = {model: 0 for model in models}
    for epoch in range(epochs):
        for agent, agent_data in data.items():
            model = models[agent]
            optimizer = optimizers[agent]
            for obs, action, reward, done, old_log_prob, value in agent_data:
                _, new_value = model(obs)
                advantage = reward + gamma * new_value * (1 - int(done)) - value
                new_log_prob = model(obs)[0]

                # TODO check the loss calculation details here
                ratio = (new_log_prob - old_log_prob).exp()
                surr1 = ratio * advantage.detach()
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (reward + gamma * new_value.detach() * (1 - int(done)) - value).pow(2).mean()
                acc_losses[agent] += (policy_loss.detach().item() + value_loss.detach().item())

                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                optimizer.step()
    return {agent: acc_losses[agent] / (epochs * len(data)) for agent in acc_losses}


env = color_maze.ColorMaze()
# Observation and action spaces are the same for leader and follower
obs_space = env.observation_space('leader')
act_space = env.action_space('leader')

SAVE_DATA = False

leader = ActorCritic(obs_space, act_space)
follower = ActorCritic(obs_space, act_space)
leader_optimizer = optim.Adam(leader.parameters(), lr=0.01)
follower_optimizer = optim.Adam(follower.parameters(), lr=0.01)

num_epochs = 100
num_steps_per_epoch = 1000
ppo_epochs = 4
gamma = 0.99
clip_param = 0.2

models = {'leader': leader, 'follower': follower}
optimizers = {'leader': leader_optimizer, 'follower': follower_optimizer}

for epoch in range(num_epochs):
    data = collect_data(env, models, num_steps_per_epoch)
    if SAVE_DATA:
        # TODO serialize the episode trajectory for future use
        pass

    losses = ppo_update(models, optimizers, data, ppo_epochs, gamma, clip_param)
    print(losses)
