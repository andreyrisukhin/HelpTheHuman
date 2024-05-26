import heapq
import numpy as np
import color_maze
from color_maze import Moves, IDs
import torch

class AStarAgent:
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]
        self.show_obs = show_obs
        self.path = None

    def __call__(self, observation, agent):
        # only trigger when we are the correct agent
        assert (
            agent == self.agent
        ), f"A* Policy only applied to agent: {self.agent}, but got tag for {agent}."

        # if the path is empty or the goal color just switched or the agent just collected a block
        # breakpoint()
        # TODO: Define self.env.goal_block_prev and self.env.leader.collected_block in ColorMaze or determine way to
        # pass information into a_star_policy and make methods to handle it
        if not self.path or self.env.goal_block != self.env.goal_block_prev or self.env.leader.collected_block:
            self.path = self.a_star_search()
            breakpoint()
            self.env.goal_block_prev = self.env.goal_block
            assert self.path, "No path found"

        return self.get_next_action(self.path)

    def heuristic(self, leader_pos, goal_pos):
        """
        Manhattan distance as the heuristic function
        """
        return torch.sum(torch.abs(leader_pos - goal_pos), dim=-1)

    def a_star_search(self):
        """
        Performs A* search to find the shortest path for the leader to the closest correct block
        """
        blocks = self.env.blocks
        start_pos = torch.tensor([self.env.leader.x, self.env.leader.y], device=self.env.device)
        print(start_pos)
        goal_block = self.env.goal_block.value
        goal_positions = torch.nonzero(blocks[goal_block] == 1, as_tuple=True)

        # Inv: There should always be correct blocks
        if len(goal_positions[0]) == 0:
            assert False, "No correct blocks found"

        # Calculate the closest correct block to the leader
        goal_positions_tensor = torch.stack(goal_positions, dim=1)
        goal_dists = self.heuristic(start_pos.repeat(goal_positions_tensor.shape[0], 1), goal_positions_tensor)
        closest_goal_idx = torch.argmin(goal_dists)
        goal_pos = goal_positions_tensor[closest_goal_idx]

        # Initialize the open and closed sets
        open_set = [(0 + self.heuristic(start_pos, goal_pos).item(), start_pos.tolist(), [])]
        heapq.heapify(open_set)
        closed_set = set()

        # Loop until the open set is empty or the goal is reached
        while open_set:
            current_cost, current_pos, current_path = heapq.heappop(open_set)

            # If the goal is reached, return the path
            if current_pos == list(goal_pos.tolist()):
                return current_path

            closed_set.add(tuple(current_pos))

            # Generate the neighbors
            neighbors = [
                (current_pos[0], current_pos[1] + 1),
                (current_pos[0], current_pos[1] - 1),
                (current_pos[0] + 1, current_pos[1]),
                (current_pos[0] - 1, current_pos[1]),
            ]

            for neighbor in neighbors:
                neighbor_pos = torch.tensor(neighbor, dtype=torch.long, device=self.env.device)
                # Check if the neighbor is within the boundaries and not a wall
                if self.env.leader.is_legal(neighbor_pos[0].item(), neighbor_pos[1].item()):
                    new_cost = current_cost - self.heuristic(torch.tensor(current_pos, device=self.env.device), goal_pos).item() + 1 + self.heuristic(neighbor_pos, goal_pos).item()
                    new_path = current_path + [tuple(neighbor_pos.tolist())]

                    # Check if the neighbor is in the closed set or the open set
                    if tuple(neighbor_pos.tolist()) in closed_set:
                        continue
                    in_open_set = False
                    for idx, entry in enumerate(open_set):
                        if entry[1] == neighbor_pos.tolist():
                            in_open_set = True
                            if entry[0] > new_cost:
                                open_set[idx] = (new_cost, neighbor_pos.tolist(), new_path)
                            break

                    if not in_open_set:
                        heapq.heappush(open_set, (new_cost, neighbor_pos.tolist(), new_path))

        # Inv: Path should never be empty
        assert False, "No path found"

    def get_next_action(self, path):
        """
        Converts the path into a sequence of actions
        """
        actions = []
        # if path only has one element, return that element
        if len(path) == 1:
            x1, y1 = path[0]
            x2, y2 = self.env.leader.x, self.env.leader.y
            if x2 > x1:
                return Moves.RIGHT.value
            elif x2 < x1:
                return Moves.LEFT.value
            elif y2 > y1:
                return Moves.DOWN.value
            else:
                return Moves.UP.value
        
        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            breakpoint()
            if x2 > x1:
                actions.append(Moves.RIGHT.value)
            elif x2 < x1:
                actions.append(Moves.LEFT.value)
            elif y2 > y1:
                actions.append(Moves.DOWN.value)
            else:
                actions.append(Moves.UP.value)

        self.actions = actions
        
        return actions[0]
    