from pathlib import Path

import gym
import d4rl
import numpy as np
import torch
from tqdm import trange

from src_iql_gwthomas.iql import ImplicitQLearning
from src_iql_gwthomas.policy import GaussianPolicy, DeterministicPolicy
from src_iql_gwthomas.value_functions import TwinQ, ValueFunction
from src_iql_gwthomas.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy

from color_maze import ColorMaze

"""
Notes about IQL

QL is optimistic in face of uncertainty: taking max over noise is overoptimistic 
* Q_policy () = ..<s,a>. + discount * argmax_a' (Q_target(a',s))
* If there are four distributional measurements, and I've observed a3 more than a4, a4 will be noisier -> could be greater than a3
* This is because we are estimating value | observed values. Observing action a changes the distribution of Expected[value_a]

To compensate, offline RL is pessimistic: Q_target updates slower than Q_policy, 100-1000x, and updates to approach Q_policy.
This works empirically, and 10yrs later due to avoiding rank collapse (1000x slower means Qtarget and Qpolicy are different enough)
https://arxiv.org/abs/2010.14498 

Replay buffer should store tuples (s, a, r, s') for IQL. // no discount or argmax around Q(a',s) with buffer (s, a, r, s', a') for CQL.

Use IQL

Andrey: confused what this means, use a frozen model to collect data? And use that "expert" data for q-learning?

"""

""" Sriyash OH 5-3-2024
ensure that the behavior we want to extract is represented in the dataset, offline RL will not do anything novel
    can be collection or 1, but must be covered
    Reward labels are also needed

Start with 1M timesteps, this is standard (especially since we are not using img channels)
    Use the Jax implementation, blazing fast (Natasha also mentioned)
        do 2 things
        1) create offline RL env the way it wants us to create. Write conv net ourselves, because Jax has code for flattened vectors. Use the encoders from https://github.com/dibyaghosh/jaxrl_m/blob/main/jaxrl_m/vision/small_encoders.py

    1k timesteps, 20 min

    Check website for compute access via class

    dr4l stores as hdf5 file, dict of their info. Flattened vec, 1M x 32 x 32 x 5. Be careful storing terminals! Next obs in d4rl is sometimes the first obs of next env, if terminated. They store with an offset [0, -1] [1, end]
        1M time steps: extract obs. They index [0, 999999] as obs for first. Then index [100, 1M]. The env assumes we mask out last obs anyway

"""

def get_env_and_dataset(log, env_name, max_episode_steps):
    """
    Should we instead be loading checkpoints from a frozen data collection run?
    'Each task is associated with a fixed offline dataset, which can be obtained with the env.get_dataset() method' implies we should create a dataset from an env and fully trained policy. Does that sound right?
    """


    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    our_env = ColorMaze() # TODO check init params, set seed here instead of in main()?
    our_dataset = our_env.get_qlearnng_dataset()

    # TODO replace with our env. env.get_dataset() -> observations, actions, rewards, terminals, timeouts, infos.
    # .qlearning_dataset() also returns a next_observations

    # Replay buffer (s,a,r,s') (observations, actions, rewards, next_observations)

    if any(s in env_name for s in ('halfcheetah', 'hopper', 'walker2d')):
        min_ret, max_ret = return_range(dataset, max_episode_steps)
        log(f'Dataset returns have range [{min_ret}, {max_ret}]')
        dataset['rewards'] /= (max_ret - min_ret)
        dataset['rewards'] *= max_episode_steps
    elif 'antmaze' in env_name:
        dataset['rewards'] -= 1.

    for k, v in dataset.items():
        dataset[k] = torchify(v)

    return env, dataset


def main(args):
    torch.set_num_threads(1)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')

    env, dataset = get_env_and_dataset(log, args.env_name, args.max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)

    # if args.deterministic_policy:
    #     policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    # else:
    #     policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
    policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, args.max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })

    iql = ImplicitQLearning(
        qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        max_steps=args.n_steps,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount
    )

    for step in trange(args.n_steps):
        iql.update(**sample_batch(dataset, args.batch_size))
        if (step+1) % args.eval_period == 0:
            eval_policy()

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=10**6)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=5000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    main(parser.parse_args())