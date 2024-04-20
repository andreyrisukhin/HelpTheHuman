import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Any, Mapping
from dataclasses import dataclass
import wandb
from fire import Fire
from tqdm import tqdm
import os


from replay_trajectory import replay_trajectory

from src import color_maze

@dataclass
class StepData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    loss: float
    explained_var: float


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Network structure from "Emergent Social Learning via Multi-agent Reinforcement Learning": https://arxiv.org/abs/2010.00581
        self.shared_network = nn.Sequential(
            layer_init(nn.Conv2d(observation_space.shape[0], 32, kernel_size=3, stride=3, padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),  # flatten all dims except batch-wise
            layer_init(nn.Linear(64*6*6, 192)),
            nn.Tanh(),
            # No history is included in observations for now, so LSTM doesn't make sense
            # nn.LSTM(192, 192, batch_first=True),
            layer_init(nn.Linear(192, 192))
        )
        self.policy_network = nn.Sequential(
            layer_init(nn.Linear(192, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space.n), std=0.01),
        )
        self.value_network = nn.Sequential(
            layer_init(nn.Linear(192, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        # Removing hidden states and cells because LSTM is replaced by Linear for now
        # features, (hidden_states, cell_states) = self.shared_network(x)
        features = self.shared_network(x)
        return self.policy_network(features), self.value_network(features)


    def get_value(self, x):
        return self.value_network(self.shared_network(x))


    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value


def step(
        envs: list[color_maze.ColorMaze],
        models: Mapping[str, ActorCritic],
        optimizers: Mapping[str, optim.Optimizer],
        num_steps: int,
        batch_size: int,
        minibatch_size: int,
        gamma: float,
        gae_lambda: float,
        ppo_update_epochs: int,
        norm_advantage: bool,
        clip_param: float,
        clip_vloss: bool,
        entropy_coef: float,
        value_func_coef: float,
        max_grad_norm: float,
        target_kl: float | None,
) -> dict[str, StepData]:
    observation_space_shapes = {
        agent: envs[0].observation_space(agent).shape
        for agent in models
    }
    observation_space_shapes = {key: value for key, value in observation_space_shapes.items() if value is not None}
    assert len(observation_space_shapes) == len(models)
    action_space_shapes = {
        agent: envs[0].action_space(agent).shape
        for agent in models
    }
    action_space_shapes = {key: value for key, value in action_space_shapes.items() if value is not None}
    assert len(action_space_shapes) == len(models)

    all_observations = {agent: torch.zeros((num_steps, len(envs)) + observation_space_shapes[agent]).to(DEVICE) for agent in models}
    all_actions = {agent: torch.zeros((num_steps, len(envs)) + action_space_shapes[agent]).to(DEVICE) for agent in models}
    all_logprobs = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}
    all_rewards = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}
    all_dones = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}
    all_values = {agent: torch.zeros((num_steps, len(envs))).to(DEVICE) for agent in models}

    next_observation_dicts, _ = list(zip(*[env.reset() for env in envs]))
    next_observations = {agent: np.array([obs_dict[agent] for obs_dict in next_observation_dicts]) for agent in models}
    next_observations = {agent: torch.tensor(next_observations[agent]).to(DEVICE) for agent in models}
    next_dones = {agent: torch.zeros(len(envs)).to(DEVICE) for agent in models}

    for step in range(num_steps):
        step_actions = {}

        for agent, model in models.items():
            all_observations[agent][step] = next_observations[agent]
            all_dones[agent][step] = next_dones[agent]

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_observations[agent])
                step_actions[agent] = action.cpu().numpy()

                all_actions[agent][step] = action
                all_logprobs[agent][step] = logprob
                all_values[agent][step] = value.flatten()

        # Convert step_actions from dict of lists to list of dicts
        step_actions = [{agent: step_actions[agent][i] for agent in step_actions} for i in range(len(step_actions[list(models.keys())[0]]))]

        next_observation_dicts, reward_dicts, terminated_dicts, truncation_dicts, _ = list(zip(*[env.step(step_actions[i]) for i, env in enumerate(envs)]))
        next_observations = {agent: np.array([obs_dict[agent] for obs_dict in next_observation_dicts]) for agent in models}
        rewards = {agent: np.array([reward_dict[agent] for reward_dict in reward_dicts]) for agent in models}
        for agent in models:
            all_rewards[agent][step] = torch.tensor(rewards[agent]).to(DEVICE).view(-1)
        next_dones = {agent: np.logical_or([int(terminated[agent]) for terminated in terminated_dicts], [int(truncated[agent]) for truncated in truncation_dicts]) for agent in models}

        next_observations = {agent: torch.tensor(next_observations[agent]).to(DEVICE) for agent in models}
        next_dones = {agent: torch.tensor(next_dones[agent], dtype=torch.float32).to(DEVICE) for agent in models}

    explained_var = {}
    acc_losses = {agent: 0 for agent in models}
    for agent, model in models.items():
        # bootstrap values if not done
        with torch.no_grad():
            next_values = model.get_value(next_observations[agent]).reshape(1, -1)
            advantages = torch.zeros_like(all_rewards[agent]).to(DEVICE)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1 - next_dones[agent]
                    nextvalues = next_values
                else:
                    nextnonterminal = 1 - all_dones[agent][t + 1]
                    nextvalues = all_values[agent][t + 1]
                delta = all_rewards[agent][t] + gamma * nextvalues * nextnonterminal - all_values[agent][t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + all_values[agent]

        # flatten the batch
        b_obs = all_observations[agent].reshape((-1,) + observation_space_shapes[agent])
        b_logprobs = all_logprobs[agent].reshape(-1)
        b_actions = all_actions[agent].reshape((-1,) + action_space_shapes[agent])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = all_values[agent].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(ppo_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_param).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_advantage:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_param,
                        clip_param,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - entropy_coef * entropy_loss + v_loss * value_func_coef
                acc_losses[agent] += loss.detach().cpu().item()

                optimizers[agent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizers[agent].step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var[agent] = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    step_result = {
        agent: StepData(
            observations=all_observations[agent].cpu(),
            actions=all_actions[agent].cpu(),
            rewards=all_rewards[agent].cpu(),
            dones=all_dones[agent].cpu(),
            action_log_probs=all_logprobs[agent].cpu(),
            values=all_values[agent].cpu(),
            loss=acc_losses[agent] / ppo_update_epochs,
            explained_var=explained_var[agent]
        )
        for agent in models
    }
    return step_result


def train(
        run_name: str | None = None,
        # PPO params
        total_timesteps: int = 500000,
        learning_rate: float = 1e-4,  # default set from "Emergent Social Learning via Multi-agent Reinforcement Learning"
        num_envs: int = 4,
        num_steps_per_rollout: int = 128,
        gamma: float = 0.99,  # discount factor
        gae_lambda: float = 0.95,  # lambda for general advantage estimation
        num_minibatches: int = 4,
        ppo_update_epochs: int = 4,
        norm_advantage: bool = True,  # toggle advantage normalization
        clip_param: float = 0.2,  # surrogate clipping coefficient
        clip_vloss: bool = True,  # toggle clipped loss for value function
        entropy_coef: float = 0.01,
        value_func_coef: float = 0.5,
        max_grad_norm: float = 0.5,  # max gradnorm for gradient clipping
        target_kl: float | None = None,  # target KL divergence threshold
        # Config params
        save_data_iters: int = 100,
        checkpoint_iters: int = 0,
        debug_print: bool = False,
        log_to_wandb: bool = True,
        seed: int = 42,
):
    if log_to_wandb:
        wandb.init(entity='kavel', project='help-the-human', name=run_name)
    os.makedirs(f'results/{run_name}', exist_ok=True)

    torch.manual_seed(seed)

    batch_size = num_envs * num_steps_per_rollout
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    envs = [color_maze.ColorMaze() for _ in range(num_envs)]

    # Observation and action spaces are the same for leader and follower
    obs_space = envs[0].observation_space('leader')
    act_space = envs[0].action_space('leader')

    leader = ActorCritic(obs_space, act_space).to(DEVICE)
    follower = ActorCritic(obs_space, act_space).to(DEVICE)
    leader_optimizer = optim.Adam(leader.parameters(), lr=learning_rate, eps=1e-5)
    follower_optimizer = optim.Adam(follower.parameters(), lr=learning_rate, eps=1e-5)
    models = {'leader': leader, 'follower': follower}
    optimizers = {'leader': leader_optimizer, 'follower': follower_optimizer}

    print(f'Running for {num_iterations} iterations using {num_envs} envs with {batch_size=} and {minibatch_size=}')

    for iteration in tqdm(range(num_iterations), total=num_iterations):
        step_results = step(
            envs=envs,
            models=models,
            optimizers=optimizers,
            num_steps=num_steps_per_rollout,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_update_epochs=ppo_update_epochs,
            norm_advantage=norm_advantage,
            clip_param=clip_param,
            clip_vloss=clip_vloss,
            entropy_coef=entropy_coef,
            value_func_coef=value_func_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl
        )


        metrics = {'leader': {}, 'follower': {}}
        metrics['leader']['loss'] = step_results['leader'].loss
        metrics['follower']['loss'] = step_results['follower'].loss
        metrics['leader']['explained_var'] = step_results['leader'].explained_var
        metrics['follower']['explained_var'] = step_results['follower'].explained_var
        metrics['leader']['reward'] = step_results['leader'].rewards.sum(dim=0).mean()  # Sum along step dim and average along env dim
        metrics['follower']['reward'] = step_results['follower'].rewards.sum(dim=0).mean()  # Sum along step dim and average along env dim
        if log_to_wandb:
            wandb.log(metrics, step=iteration)

        if save_data_iters and iteration % save_data_iters == 0:
            observation_states = step_results['leader'].observations.transpose(0, 1)  # Transpose so the dims are (env, step, ...observation_shape)
            for i in range(observation_states.size(0)):
                trajectory = observation_states[i].numpy()
                os.makedirs(f'trajectories/{run_name}', exist_ok=True)
                np.save(f"trajectories/{run_name}/trajectory_{iteration=}_env={i}.npy", trajectory)

        if debug_print:
            print(f"iter {iteration}: {metrics}")

        if checkpoint_iters and iteration % checkpoint_iters == 0:
            print(f"Saving models at epoch {iteration}")
            torch.save(leader.state_dict(), f'results/{run_name}/leader_{iteration=}.pth')
            torch.save(follower.state_dict(), f'results/{run_name}/follower_{iteration=}.pth')

    torch.save(leader.state_dict(), f'results/{run_name}/leader_{iteration=}.pth')
    torch.save(follower.state_dict(), f'results/{run_name}/follower_{iteration=}.pth')


if __name__ == '__main__':
    Fire(train)
