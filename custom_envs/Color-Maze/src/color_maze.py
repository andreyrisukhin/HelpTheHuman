"""
Two players: a leader, and a follower. Multiple colors of blocks.
The leader moves into blocks of a particular color to score.
The follower can do the same. If the follower moves into a different color, score decreases (visible?)

Spawn: leader in top left, follower in bottom right. Blocks in random locations, not over other entities.
Movement: leader and follower share moveset, one grid up, down, left, right.
"""
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box # Fundamental Spaces - https://gymnasium.farama.org/api/spaces/
from gymnasium.spaces import Dict # Composite Spaces - Dict is best for fixed number of unordered spaces.

from pettingzoo import ParallelEnv

from typing import List, Callable, Dict
from dataclasses import dataclass
from enum import Enum
class Moves(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
class Boundary(Enum):
    # Invariant: bounds are inclusive
    x1 = 0
    y1 = 0
    x2 = 31
    y2 = 31
xBoundary = Boundary.x2.value + 1 - Boundary.x1.value
yBoundary = Boundary.y2.value + 1 - Boundary.y1.value
class IDs(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2
    LEADER = 3
    FOLLOWER = 4

@dataclass 
class Agent:
    '''Agent class to store x and y coordinates of the agent. Automatically creates __init__ and __repr__ methods.'''
    x: int
    y: int

class ColorMazeRewards():
    '''Class to organize reward functions for the ColorMaze environment.
    
    tldr: 
        To define new reward_shaping_function, go to color_maze.ColorMazeRewards and add it there. 
        To use them, go to run_ppo.py and when initializing a ColorMaze() environment, init as ColorMaze(reward_shaping_fns=[REWARD_SHAPINGS_HERE])
    
    Invariant [!]: All reward shaping functions take args: agents dictionary, rewards dictionary; and return the rewards dictionary.'''

    def __init__(self, close_threshold:int=10, timestep_expiry:int=500) -> None:
        self.close_threshold = close_threshold
        self.timestep_expiry = timestep_expiry

    def penalize_follower_close_to_leader(self, agents:Dict[str, Agent], rewards, step:int):
        '''Penalize the follower if it is close to the leader.'''
        if step > self.timestep_expiry:
            return rewards
        leader = agents["leader"]
        follower = agents["follower"]
        if abs(leader.x - follower.x) + abs(leader.y - follower.y) < self.close_threshold:
            rewards["follower"] -= 1 # TODO could shape this curve, rather than Relu.
        return rewards

class ColorMaze(ParallelEnv):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name:": "color_maze_v0",
    }

    def __init__(self, seed=None, reward_shaping_fns:List[Callable]=[]):
        """Initializes the environment's random seed and sets up the environment.

        reward_shaping_fns: List of reward shaping function to be applied. The caller will need to import ColorMazeRewards and pass the functions from here.
        """

        # Randomness
        self.seed = seed  # For inspection
        if self.seed is None:
            self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)
        
        self.goal_block = IDs.RED
        self.prob_block_switch = 0.01 # Uniformly at random, expect 1 switch every 100 timesteps.

        # Agents
        self.possible_agents = ["leader", "follower"]
        self.leader = Agent(Boundary.x1.value, Boundary.y1.value)
        self.follower = Agent(Boundary.x2.value, Boundary.y2.value)
        self._action_space = Discrete(4)  # Moves: Up, Down, Left, Right

        # Blocks - invariant: for all (x, y) coordinates, no two slices are non-zero
        self.blocks = np.zeros((3, xBoundary, yBoundary))
        self._n_channels = self.blocks.shape[0] + len(self.possible_agents)  # 5: 1 channel for each block color + 1 for each agent
        board_space = Box(low=0, high=1, shape=(self._n_channels, xBoundary, yBoundary), dtype=np.int32)
        goal_block_space = Discrete(3)  # Red, Green, Blue
        self._observation_space = board_space # TODO add history of leader information
        observation_space_with_goal = dict({
            "observation": board_space,
            "goal_block": goal_block_space # TODO ensure this is only visible to the leader, follower should have this either absent or masked out.
        })

        # Spaces
        self.observation_spaces = {
            agent: self._observation_space
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: self._action_space
            for agent in self.possible_agents
        }

        self.observation_space = lambda agent: self._observation_space
        self.action_space = lambda agent: self._action_space

        # Environment duration
        self.timestep = 0
        self._MAX_TIMESTEPS = 1000

        # Reward shaping
        self.reward_shaping_fns = reward_shaping_fns

    def _randomize_goal_block(self): 
        if self.rng.random() < self.prob_block_switch:
            random_idx = self.rng.integers(len([IDs.RED, IDs.GREEN, IDs.BLUE])) # We only want to switch between the 3 colors, not the other spaces.
            self.goal_block = IDs(random_idx)

    def _convert_to_observation(self):
        """
        Converts the internal state of the environment into an observation that can be used by the agent.
        
        The observation is a 3D numpy array where each cell (0 or 1) represents the presence of an object in that cell in the maze.
        The dimensions are based on IDs values:
        - 0: Red block
        - 1: Blue block
        - 2: Green block
        - 3: Leader agent
        - 4: Follower agent

        Returns:
            numpy.ndarray: The observation array.
        """
        leader_position = np.zeros((1, xBoundary, yBoundary))
        follower_position = np.zeros((1, xBoundary, yBoundary))
        leader_position[0, self.leader.x, self.leader.y] = 1
        follower_position[0, self.follower.x, self.follower.y] = 1

        observation = np.concatenate((self.blocks, leader_position, follower_position), axis=0)

        # Ensure that observation is a 2d array
        assert observation.ndim == 3
        assert observation.shape == (self._n_channels, xBoundary, yBoundary)
        return observation.astype(np.float32)

    def set_state_to_observation(self, observation: np.ndarray):
        """
        Converts the format returned from _convert_to_observation
        into the internal env state representation.
        *Overrides* [!] the current env state with the given observation.
        """
        # Add 1 to IDs because 0 is empty space
        leader_places = observation[IDs.LEADER.value].reshape((xBoundary, yBoundary))
        follower_places = observation[IDs.FOLLOWER.value].reshape((xBoundary, yBoundary))
        assert leader_places.sum() == 1
        assert follower_places.sum() == 1
        self.blocks = observation[IDs.RED.value : IDs.GREEN.value + 1]
        assert self.blocks.shape == (3, xBoundary, yBoundary)
        self.leader.x, self.leader.y = np.argwhere(leader_places).flatten()
        self.follower.x, self.follower.y = np.argwhere(follower_places).flatten()

    def reset(self, *, seed=None, options=None):
        """Reset the environment to a starting point."""
        if seed is not None:
            self.seed = seed
        else:
            self.seed = 42
        self.rng = np.random.default_rng(seed=self.seed)

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        # Randomize initial locations
        self.leader.x = self.rng.integers(Boundary.x1.value, Boundary.x2.value, endpoint=True)
        self.leader.y = self.rng.integers(Boundary.y1.value, Boundary.y2.value, endpoint=True)
        self.follower.x = self.leader.x
        self.follower.y = self.leader.y
        while (self.follower.x, self.follower.y) == (self.leader.x, self.leader.y):
            self.follower.x = self.rng.integers(Boundary.x1.value, Boundary.x2.value, endpoint=True)
            self.follower.y = self.rng.integers(Boundary.y1.value, Boundary.y2.value, endpoint=True)

        self.blocks = np.zeros((3, xBoundary, yBoundary))

        # Randomly place 5% blocks (in a 31x31, 16 blocks of each color)
        for _ in range(16):
            self._consume_and_spawn_block(IDs.RED.value, 0, 0)
            self._consume_and_spawn_block(IDs.GREEN.value, 0, 0)
            self._consume_and_spawn_block(IDs.BLUE.value, 0, 0)
        
        self._randomize_goal_block()

        observation = self._convert_to_observation()
        observations = {
            agent: observation
            for agent in self.agents
        }

        # Get dummy info, necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _consume_and_spawn_block(self, color_idx:int, x:int, y:int) -> None:
        self.blocks[color_idx, x, y] = 0
        # Find a different cell that is not occupied (leader, follower, existing block) and set it to this block.
        # Also make sure no other color is present there      
        zero_indices = np.argwhere(np.all((self.blocks == 0), axis=0))
        self.rng.shuffle(zero_indices)
        for x,y in zero_indices:
            if ((x == self.leader.x and y == self.leader.y) or
                (x == self.follower.x and y == self.follower.y)):
                continue

            self.blocks[color_idx, x, y] = 1
            return
        assert False, "No cell with value 0 found to update."

    def step(self, actions):
        """
        Takes an action for all agents in environment, and assigns rewards.
        """
        leader_action = actions["leader"]
        follower_action = actions["follower"]

        def _move(x, y, action):
            """
            Always call _move for the leader first in a given timestep. The leader is favored in collisions with follower. 
            """
            new_x, new_y = x, y
            if action == Moves.UP.value and y < Boundary.y2.value:
                new_y += 1
            elif action == Moves.DOWN.value and y > Boundary.y1.value:
                new_y -= 1
            elif action == Moves.LEFT.value and x > Boundary.x1.value:
                new_x -= 1
            elif action == Moves.RIGHT.value and x < Boundary.x2.value:
                new_x += 1
    
            if (new_x, new_y) == (self.leader.x, self.leader.y):
                return x, y
            else:
                return new_x, new_y
        
        self.leader.x, self.leader.y = _move(self.leader.x, self.leader.y, leader_action)
        self.follower.x, self.follower.y = _move(self.follower.x, self.follower.y, follower_action)

        self._randomize_goal_block()

        # Make action masks
        leader_action_mask = np.ones(4)
        follower_action_mask = np.ones(4)
        for action_mask, x, y in zip([leader_action_mask, follower_action_mask], [self.leader.x, self.follower.x], [self.leader.y, self.follower.y]):
            if x == Boundary.x1.value:
                action_mask[Moves.LEFT.value] = 0  # cant go left
            if x == Boundary.x2.value:
                action_mask[Moves.RIGHT.value] = 0  # cant go right
            if y == Boundary.y1.value:
                action_mask[Moves.DOWN.value] = 0  # cant go down
            if y == Boundary.y2.value:
                action_mask[Moves.UP.value] = 0  # cant go up

        # Give rewards
        shared_reward = 0
        for agent, x, y in zip(["leader", "follower"], [self.leader.x, self.follower.x], [self.leader.y, self.follower.y]):
            if self.blocks[self.goal_block.value, x, y]:
                shared_reward += 1
                self._consume_and_spawn_block(self.goal_block.value, x, y)
            else:
                for non_reward_block_idx in [i for i in range(self.blocks.shape[0]) if i != self.goal_block.value]:
                    if self.blocks[non_reward_block_idx, x, y]:
                        shared_reward -= 1
                        self._consume_and_spawn_block(non_reward_block_idx, x, y)
                        break # TODO ASK, why do we break here? Don't we want to process ALL non-reward blocks?

        rewards = {'leader': shared_reward, 'follower': shared_reward}

        # Apply reward shaping
        for reward_shaping_function in self.reward_shaping_fns:
            rewards = reward_shaping_function(dict({'leader': self.leader, 'follower': self.follower}), rewards, self.timestep)

        # Check termination conditions
        termination = False
        if self.timestep > self._MAX_TIMESTEPS:
            termination = True
        self.timestep += 1

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        # Formatting by agent for the return types
        terminateds = {a: termination for a in self.agents}
        if termination:
            self.agents = []

        observation = self._convert_to_observation()
        observations = {
            agent: observation
            for agent in self.agents
        }
        truncateds = terminateds
        return observations, rewards, terminateds, truncateds, infos

    def render(self):
        """Render the environment."""
        grid = np.full((Boundary.x2.value + 1, Boundary.y2.value + 1), ".")
        grid[self.leader.x, self.leader.y] = "L"
        grid[self.follower.x, self.follower.y] = "F"
        for x, y in np.argwhere(self.blocks[IDs.RED.value]):
            grid[x, y] = "R"
        for x, y in np.argwhere(self.blocks[IDs.GREEN.value]):
            grid[x, y] = "G"
        for x, y in np.argwhere(self.blocks[IDs.BLUE.value]):
            grid[x, y] = "B"

        # Flip it so y is increasing upwards
        grid = np.flipud(grid.T)
        rendered = ""
        for row in grid:
            rendered += "".join(row) + "\n"
        print(rendered)
