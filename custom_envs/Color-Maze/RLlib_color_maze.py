"""
Script to run Independent PPO on both leader and follower agents.
Inspired by this example of using RLlib with PettingZoo: https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent/rock_paper_scissors_learned_vs_learned.py
"""

import re 

from src import color_maze # The environment to train on

# run `pip install "ray[rllib]" torch` to install RLlib for pytorch
from ray.rllib.connectors.env_to_module import (
    AddObservationsFromEpisodesToBatch,
    FlattenObservations,
    WriteObservationsToEpisodes,
)
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env, get_trainable_cls

# Below and modified config from experimental api, recommended for multiagent PPO: https://docs.ray.io/en/latest/rllib/rllib-new-api-stack.html
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner
    # Example had their env, we use ours.

parser = add_rllib_example_script_args(
    default_iters=50,
    default_timesteps=200000,
    default_reward=0.6
) # TODO where do these come from?
parser.add_argument(
    "--use-lstm",
    action="store_true",
    help="""Whether to use an LSTM wrapped module instead of a simple MLP one. With LSTM 
    the reward diff can reach 7.0, without only 5.0."""
) # Same question

register_env(
    "color_maze",
    lambda _: ParallelPettingZooEnv(color_maze.ColorMaze()),
)



if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents == 2, "Must set --num-agents=2, this script only supports 2 agents"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script" # New api stack supports multi-agent PPO.

    '''
    base_config = (
        get_trainable_cls(args.algo) # PPO
        .get_default_config()
        .environment("color_maze")
        .rollouts(
            env_runner_cls=MultiAgentEnvRunner,
            # env_to_module_connector=lambda env: (
            #     AddObservationsFromEpisodesToBatch(),
            #     FlattenObservations(multi_agent=True),
            #     WriteObservationsToEpisodes(),
            # ),
        )
        .multi_agent(
            policies={"leader", "follower"},
            policy_mapping_fn = lambda agent_id, episode: agent_id,
        ) # Shouldn't we add .framework("torch"), .evaluation() here?
        .training(
            model={
                "use_lstm": args.use_lstm,
                # Use a simpler FCNet when we also have an LSTM.
                "fcnet_hiddens": [32] if args.use_lstm else [256,256], # Why do these shapes make sense?
                "lstm_cell_size": 256,
                "max_seq_len": 15, # Where do these come from?
                "vf_share_layers": True, # What does this do?
                "uses_new_env_runners": True,
            },
            vf_loss_coeff=0.005,
        )
        # .rl_module(
        #     rl_module_spec=MultiAgentRLModuleSpec(
        #         module_specs={
        #             "leader": SingleAgentRLModuleSpec(),
        #             "follower": SingleAgentRLModuleSpec(),
        #         }
        #     )
        # )
        .experimental(_enable_new_api_stack=True)
    )
    '''
    config = (
        PPOConfig().environment(color_maze.ColorMaze)
        .experimental(_enable_new_api_stack=True)
        .rollouts(env_runner_cls=MultiAgentEnvRunner)
        .resources(
            num_learner_workers=1, # Set this to the number of GPUs 
            num_gpus_per_learner_worker=1, # 1 often, if at least one GPU
            num_cpus_for_local_worker=1,
        )
        .training(model={"uses_new_env_runners": True})
        .multi_agent(
            policies={"leader", "follower"},
            policy_mapping_fn=lambda agent_id, episode: agent_id,
        )
    )

    run_rllib_example_script_experiment(config, args)
