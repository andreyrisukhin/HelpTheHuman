"""
Two players: a leader, and a follower. Multiple colors of blocks.
The leader moves into blocks of a particular color to score.
The follower can do the same. If the follower moves into a different color, score decreases (visible?)

Spawn: leader in top left, follower in bottom right. Blocks in random locations, not over other entities.
Movement: leader and follower share moveset, one grid up, down, left, right.
"""

import functools
import random 
from copy import copy 

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

from enum import Enum
class Moves(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
class Boundary(Enum):
    x1 = 0
    y1 = 0
    x2 = 6
    y2 = 6


class ColorMaze(ParallelEnv):
    """The metadata holds environment constants.
    
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {
        "name:": "color_maze_v0",
    }

    def __init__(self):
        """The init method takes in environment arguments.
        
        Defines the following attributes:
        - possible agents (leader, follower)
        - leader coordinates (x,y)
        - follower coordinates (x,y)
        - red blocks [(x,y), (x,y), ...]
        - green blocks [(x,y), (x,y), ...]
        - blue blocks [(x,y), (x,y), ...]
        - timestep

        Spaces are defined in the action_space and observation_space methods.
        If not overridden, spaces are inferred from self.observation_space and self.action_space.
        """

        self.possible_agents = ["leader", "follower"]
        self.leader_x = None
        self.leader_y = None
        self.follower_x = None
        self.follower_y = None
        self.red_blocks = []
        self.green_blocks = []
        self.blue_blocks = []
        self.timestep = None

    def reset(self, seed=None, options=None):
        """Reset the environment to a starting point.
        
        """

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.leader_x = Boundary.x1
        self.leader_y = Boundary.y1
        self.follower_x = Boundary.x2
        self.follower_y = Boundary.y2
        self.red_blocks = [(1,0), (2,1)] # TODO random generate
        self.green_blocks = [(0,1), (1,2)]
        self.blue_blocks = [(2,0), (2,2)]

        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
            # What is this for??
            # Assuming this is the visible range of the agent
        }

        # Get dummy info, necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        """
        Takes an action for current agent (specified by agent selection)

        Update:
        - timestep
        - infos
        - rewards
        - leader x and y
        - follower x and y
        - terminations (WHAT?)
        - truncations (WHAT?)
        - Any internal state used by observe() or render()
        """
        leader_action = actions["leader"]
        follower_action = actions["follower"]

        def _move(x, y, action):
            if action == Moves.UP and y > Boundary.y2:
                y += 1
            elif action == Moves.DOWN and y < Boundary.y1:
                y -= 1
            elif action == Moves.LEFT and x > Boundary.x1:
                x -= 1
            elif action == Moves.RIGHT and x < Boundary.x2:
                x += 1
            return x, y
        
        self.leader_x, self.leader_y = _move(self.leader_x, self.leader_y, leader_action)
        self.follower_x, self.follower_y = _move(self.follower_x, self.follower_y, follower_action)

        # Check termination conditions
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"leader": 0, "follower": 0}
            truncations = {"prisoner": True, "guard": True, "escape": True} # How do I adapt this??
        self.timestep += 1

        # Get observations
        observations = {
            a: (
                self.leader_x + 7 * self.leader_y,
                self.follower_x + 7 * self.follower_y,
            )
            for a in self.agents
        }
        # HOW TO ADAPT THIS??

        # Get dummy infos (not used in this example)
        infos = {a: {} for a in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, truncations, terminations, infos

    def render(self):
        """Render the environment."""
        grid = np.full((Boundary.x2, Boundary.y2), " ")
        grid[self.leader_x, self.leader_y] = "H"
        grid[self.follower_x, self.follower_y] = "A" # TODO rename
        for x, y in self.red_blocks:
            grid[x, y] = "R"
        for x, y in self.green_blocks:
            grid[x, y] = "G"
        for x, y in self.blue_blocks:
            grid[x, y] = "B"
        print(grid)

    # TODO continue
    def observation_space(self, agent) -> Space:
        return self.observation_space(agent)
    
    def action_space(self, agent) -> Space:
        return self.action_space(agent) # Should this be super. or self. ?