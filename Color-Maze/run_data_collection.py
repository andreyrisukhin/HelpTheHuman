"""
This file is for running a frozen model to collect data for offline RL. Save the data in a hdf5 file (standard for d4rl).
"""

# 1) Load the checkpoint
# 2) Run it in environments, like in ppo 
# 3) Save each environment to a hdf5 file

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Mapping, Sequence, Tuple
from dataclasses import dataclass
import wandb
from fire import Fire
from tqdm import tqdm
import os
from pettingzoo import ParallelEnv

from color_maze import ColorMaze
from color_maze import ColorMazeRewards
from run_ppo import ActorCritic

import h5py

@dataclass
class StepData:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    individual_rewards: np.ndarray
    shared_rewards: np.ndarray
    dones: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    # explained_var: float
    goal_info: torch.Tensor

def reset_data():
    return {
        'observations': [],
        'actions': [],
        'terminals': [],
        'rewards': [],
        'infos/goal': [],
        # 'infos/qpos': [], # Env-specific to the example
        # 'infos/qvel': [], # Env-specific to the example # TODO for ColorMaze, no need to store extra
    }

def append_data(data, s, a, tgt, done, rewards): # done is not a bool but a tensor because multiple envs
    data['observations'].append(s)
    data['actions'].append(a)
    data['terminals'].append(done)
    data['rewards'].append(rewards)
    data['infos/goal'].append(tgt)
    # data['infos/qpos'].append(env_data.qpos.ravel().copy())
    # data['infos/qvel'].append(env_data['qvel'])
    
def npify(data):
    # Convert all dict lists to numpy arrays
    for key in data:
        if key == 'terminal' or key == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32
        data[key] = np.array(data[key], dtype=dtype)
    """
    data['observations'].shape: (iter, steps, envs, 5, 32, 32)
    data['actions'].shape: (iter, steps, envs)
    data['terminals'].shape: (iter, steps, envs)
    data['rewards'].shape: (iter, steps, envs)
    data['infos/goal'].shape: (iter, steps, envs, 3)
    """
    for key in data:
        # Flatten the iteration, env, and timestep dimensions together
        transposed = data[key].transpose((0, 2, 1) + tuple(i for i in range(3, data[key].ndim)))
        data[key] = transposed.reshape((-1,) + transposed.shape[3:])
    return data
    
def initialize_hdf5(file_name:str, agent_data, steps_stored:int|None=None):
    """ Use once at the start of the data collection. """
    with h5py.File(file_name, 'a') as f:
        for key in agent_data:
            data_shape = agent_data[key][0].shape  # Assuming agent_data[key] is a list of arrays
            f.create_dataset(key, shape=(0,) + data_shape, maxshape=(steps_stored,) + data_shape, compression='gzip')
    # return h5py.File(file_name, 'a')

def append_to_hdf5(file_name:str, data):
    """ Use to append data while collecting data.
        Takes npify'd data."""
    f = h5py.File(file_name, 'a')
    for key in data:
        dataset = f[key]
        assert isinstance(dataset, h5py.Dataset), f"Expected dataset, got {type(dataset)}" # Mostly to keep typechecker happy
        new_data = np.array(data[key])
        current_size = dataset.shape[0]
        new_size = current_size + new_data.shape[0]
        dataset.resize((new_size,) + dataset.shape[1:])
        dataset[current_size:new_size] = new_data
    f.close()


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def step(
        envs: Sequence[ParallelEnv],
        models: Mapping[str, ActorCritic],
        # optimizers: Mapping[str, optim.Optimizer],
        num_steps: int,
        batch_size: int,
        minibatch_size: int,
        seeds: list[int],
        share_observation_tensors: bool = True
) -> Tuple[dict[str, StepData], int]:
    """
    Implementation is based on https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and adapted for multi-agent
    """
    observation_space_shapes = {
        agent: envs[0].observation_spaces[agent]["observation"].shape # type: ignore
        for agent in models
    }

    goal_info_shapes = {
        agent: envs[0].observation_spaces[agent]["goal_info"].shape # type: ignore
        for agent in models
    }

    observation_space_shapes = {key: value for key, value in observation_space_shapes.items() if value is not None}
    assert len(observation_space_shapes) == len(models)
    action_space_shapes = {
        agent: envs[0].action_space.shape
        for agent in models
    }
    action_space_shapes = {key: value for key, value in action_space_shapes.items() if value is not None}
    assert len(action_space_shapes) == len(models)

    if share_observation_tensors:
        assert all(shape == list(observation_space_shapes.values())[0] for shape in observation_space_shapes.values())
        # Observations are the same, so we can share the tensors to save memory
        observations = torch.zeros((num_steps, len(envs)) + list(observation_space_shapes.values())[0]).to(DEVICE)  # shape: (128, 4) + (5, 32, 32) -> (128, 4, 5, 32, 32)
        all_observations = {agent: observations for agent in models}
    else:
        all_observations = {agent: torch.zeros((num_steps, len(envs)) + observation_space_shapes[agent]).to(models[agent].device) for agent in models}  # shape: (128, 4) + (5, 32, 32) -> (128, 4, 5, 32, 32)

    all_goal_info = {
       agent: torch.zeros((num_steps, len(envs)) + goal_info_shapes[agent]).to(models[agent].device)
       for agent in models
    }
    all_actions = {agent: torch.zeros((num_steps, len(envs)) + action_space_shapes[agent]).to(models[agent].device) for agent in models}
    all_logprobs = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_rewards = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_individual_rewards = {agent: np.zeros((num_steps, len(envs))) for agent in models}
    all_shared_rewards = {agent: np.zeros((num_steps, len(envs))) for agent in models}
    all_dones = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}
    all_values = {agent: torch.zeros((num_steps, len(envs))).to(models[agent].device) for agent in models}

    # num_steps + 1 so that indexing by step gives the *input* states at that step
    lstm_hidden_states = {agent: torch.zeros((num_steps + 1, len(envs), models[agent].lstm_hidden_size)).to(models[agent].device) for agent in models}
    lstm_cell_states = {agent: torch.zeros((num_steps + 1, len(envs), models[agent].lstm_hidden_size)).to(models[agent].device) for agent in models}

    next_observation_dicts, info_dicts = list(zip(*[env.reset(seed=seed) for env, seed in zip(envs, seeds)])) # [env1{leader:{obs:.., goal_info:..}, follower:{..}} , env2...]

    if share_observation_tensors:
        next_observation = np.array([list(obs_dict.values())[0]["observation"] for obs_dict in next_observation_dicts])
        next_observation = torch.tensor(next_observation).to(DEVICE)
        next_observations = {agent: next_observation for agent in models}
    else:
        next_observations = {
            agent: np.array([obs_dict[agent]["observation"] for obs_dict in next_observation_dicts])
            for agent in models
        }
        next_observations = {agent: torch.tensor(next_observations[agent]).to(models[agent].device) for agent in models}
    next_goal_info = {
        agent: np.array([obs_dict[agent]["goal_info"] for obs_dict in next_observation_dicts])
        for agent in models
    }
    next_goal_info = {agent: torch.tensor(next_goal_info[agent], dtype=torch.float32).to(models[agent].device) for agent in models}
    next_dones = {agent: torch.zeros(len(envs)).to(models[agent].device) for agent in models}

    next_individual_rewards = {
        agent: np.array([info_dict[agent]["individual_reward"] for info_dict in info_dicts])
        for agent in models
    }

    for step in range(num_steps):
        step_actions = {}

        for agent, model in models.items():
            all_observations[agent][step] = next_observations[agent]
            all_goal_info[agent][step] = next_goal_info[agent]
            all_dones[agent][step] = next_dones[agent]

            with torch.no_grad():
                action, logprob, _, value, _, (hidden_states, cell_states) = model.get_action_and_value(next_observations[agent], next_goal_info[agent], prev_hidden_and_cell_states=(lstm_hidden_states[agent][step], lstm_cell_states[agent][step]))
                step_actions[agent] = action.cpu().numpy()
                lstm_hidden_states[agent][step + 1] = hidden_states  # step + 1 so that indexing by step gives the *input* states at that step
                lstm_cell_states[agent][step + 1] = cell_states  # step + 1 so that indexing by step gives the *input* states at that step

                all_actions[agent][step] = action
                all_logprobs[agent][step] = logprob
                all_values[agent][step] = value.flatten()

        # Convert step_actions from dict of lists to list of dicts
        step_actions = [{agent: step_actions[agent][i] for agent in step_actions} for i in range(len(step_actions[list(models.keys())[0]]))]

        next_observation_dicts, reward_dicts, terminated_dicts, truncation_dicts, info_dicts = list(zip(*[env.step(step_actions[i]) for i, env in enumerate(envs)]))
        if any('leader' not in terminated.keys() for terminated in terminated_dicts):
            breakpoint()
        
        # convert next_observation_dics to compact representation
        # next_observation_dicts = convert_to_compact(next_observation_dicts)
        
        next_observations = {agent: np.array([obs_dict[agent]['observation'] for obs_dict in next_observation_dicts]) for agent in models}
        next_goal_info = {agent: np.array([obs_dict[agent]['goal_info'] for obs_dict in next_observation_dicts]) for agent in models}
        rewards = {agent: np.array([reward_dict[agent] for reward_dict in reward_dicts]) for agent in models}
        next_individual_rewards = {agent: np.array([info_dict[agent]["individual_reward"] for info_dict in info_dicts]) for agent in models}
        next_shared_rewards = {agent: np.array([info_dict[agent]["shared_reward"] for info_dict in info_dicts]) for agent in models}
        for agent in models:
            all_rewards[agent][step] = torch.tensor(rewards[agent]).to(models[agent].device).view(-1)
            all_individual_rewards[agent][step] = next_individual_rewards[agent].reshape(-1)
            all_shared_rewards[agent][step] = next_shared_rewards[agent].reshape(-1)

        next_dones = {agent: np.logical_or([int(terminated[agent]) for terminated in terminated_dicts], [int(truncated[agent]) for truncated in truncation_dicts]) for agent in models}
        num_goals_switched = sum(env.goal_switched for env in envs) # type: ignore
        
        # Convert to tensors
        next_observations = {agent: torch.tensor(next_observations[agent]).to(models[agent].device) for agent in models}
        next_goal_info = {agent: torch.tensor(next_goal_info[agent], dtype=torch.float32).to(models[agent].device) for agent in models}
        next_dones = {agent: torch.tensor(next_dones[agent], dtype=torch.float32).to(models[agent].device) for agent in models}

        # converts the sparse channel observation representation into a compact dense one where
        # each block color is represented by a number. So we're simply storing the (x,y) and color of each block and leader + follower
        # def convert_to_compact(observation_dicts):
                  
    for agent in models:
        # Set dones to true for last step of each env to allow concating them together
        all_dones[agent][:, -1] = 1

    step_result = {
        agent: StepData(
            observations=all_observations[agent].cpu(),
            goal_info=all_goal_info[agent].cpu(),
            actions=all_actions[agent].cpu(),
            rewards=all_rewards[agent].cpu(),
            individual_rewards=all_individual_rewards[agent],
            shared_rewards=all_shared_rewards[agent],
            dones=all_dones[agent].cpu(),
            action_log_probs=all_logprobs[agent].cpu(),
            values=all_values[agent].cpu(),
        )
        for agent in models
    }
    return step_result, num_goals_switched

# python run_data_collection.py --run_name concat_arch_exp-hinf_ir_bd10_bp4m-10m_giloss02_r128_envs16_nolstm_switch_allgoalinfo_convstride1_seed0 --resume_iter 39061 --log_to_wandb False --total_timesteps 512
def collect_data(
        run_name: str | None = None,
        resume_iter: int | None = None,  # The iteration from the run to resume. Will look for checkpoints in the folder corresponding to run_name.
        log_file_name: str | None = None,
        # PPO params
        total_timesteps: int = 10**6,
        num_envs: int = 4,
        num_steps_per_rollout: int = 128,
        num_minibatches: int = 4,
        # Config params
        debug_print: bool = False,
        log_to_wandb: bool = True,
        seed: int = 42,
):
    torch.manual_seed(seed)

    batch_size = num_envs * num_steps_per_rollout
    minibatch_size = batch_size // num_minibatches
    num_iterations = total_timesteps // batch_size

    penalty_steps = num_steps_per_rollout // 4 # 512 // 4 = 128
    env_seeds = [seed + i for i in range(num_envs)]
    envs = [ColorMaze() for _ in range(num_envs)] # To add reward shaping functions, init as ColorMaze(reward_shaping_fns=[penalize_follower_close_to_leader])

    # Observation and action spaces are the same for leader and follower
    leader_obs_space = envs[0].observation_spaces['leader']
    follower_obs_space = envs[0].observation_spaces['follower']
    act_space = envs[0].action_space

    if torch.cuda.device_count() > 1:
        model_devices = {
            'leader': 'cuda:0',
            'follower': 'cuda:1'
        }
    else:
        model_devices = {
            'leader': DEVICE,
            'follower': DEVICE
        }

    leader = ActorCritic(leader_obs_space['observation'], act_space, model_devices['leader'])  # type: ignore
    follower = ActorCritic(follower_obs_space['observation'], act_space, model_devices['follower']) # type: ignore
    models = {'leader': leader, 'follower': follower}

    if resume_iter:
        # Load checkpoint state to resume run
        for agent_name, model in models.items():
            model_path = f'results/{run_name}/{agent_name}_iteration={resume_iter}.pth'
            optimizer_path = f'results/{run_name}/{agent_name}_optimizer_iteration={resume_iter}.pth'
            state_dict = torch.load(model_path)
            patched_state_dict = {}
            for key in state_dict:
                if "_orig_mod." in key:
                    patched_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
                else:
                    patched_state_dict[key] = state_dict[key]
            model.load_state_dict(patched_state_dict)
    else:
        assert False, "Data collection must use a checkpoint to resume from, specify with resume_iter."

    print(f'Running for {num_iterations} iterations using {num_envs} envs with {batch_size=} and {minibatch_size=}')

    leader_data = reset_data()
    follower_data = reset_data()

    # Append to hdf5 files incrementally
    if not log_file_name: 
        log_file_name = f"{run_name}_{resume_iter}"
    leader_file_name = log_file_name + "_leader_testing4.hdf5"
    follower_file_name = log_file_name + "_follower_testing4.hdf5"

    for iteration in tqdm(range(num_iterations), total=num_iterations):
        step_results, num_goals_switched = step(
            envs=envs,
            models=models,
            num_steps=num_steps_per_rollout,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            seeds=env_seeds,
            share_observation_tensors=(model_devices['leader'] == model_devices['follower'])
        )
        metrics = {}
        for agent, results in step_results.items(): # type: ignore
            metrics[agent] = {
                # 'loss': results.loss,
                # 'explained_var': results.explained_var,
                'reward': results.rewards.sum(dim=0).mean(),  # Sum along step dim and average along env dim
                'individual_reward': results.individual_rewards.sum(axis=0).mean(),
                'shared_reward': results.shared_rewards.sum(axis=0).mean()  
            }
        metrics['timesteps'] = (iteration + 1) * batch_size
        metrics['num_goals_switched'] = num_goals_switched

        if log_to_wandb:
            wandb.log(metrics, step=iteration)

        if debug_print:
            print(f"iter {iteration}: {metrics}")

        for i in range(len(envs)):
            env_seeds[i] += len(envs)

        # It makes sense to split leader from follower data, because their actions are distinct. 
        append_data(leader_data, step_results['leader'].observations, step_results['leader'].actions, step_results['leader'].goal_info, step_results['leader'].dones, step_results['leader'].rewards)
        append_data(follower_data, step_results['follower'].observations, step_results['follower'].actions, step_results['follower'].goal_info, step_results['follower'].dones, step_results['follower'].rewards)

        np_leader_data = npify(leader_data)
        np_follower_data = npify(follower_data)

        # Save the data to a hdf5 file incrementally.
        if iteration % 10 == 0:
            append_to_hdf5(leader_file_name, np_leader_data)
            append_to_hdf5(follower_file_name, np_follower_data)
            leader_data = reset_data()
            follower_data = reset_data()

if __name__ == '__main__':
    Fire(collect_data)
