import heapq
import numpy as np
import color_maze
from color_maze import Agent, Moves
import torch


class AStarAgent(Agent):
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        self.env = env
        self.agent_id = agent_id
        self.agent = self.env.agents[self.agent_id]
        self.show_obs = show_obs
        self.path = None

    def heuristic(self, leader_pos, goal_pos):
        """
        Manhattan distance as the heuristic function
        """
        return torch.sum(torch.abs(leader_pos - goal_pos), dim=-1)

    def a_star_search(self, env, blocks):
        """
        Performs A* search to find the shortest path for the leader to the closest correct block
        """
        start_pos = torch.tensor([env.leader.x, env.leader.y], dtype=torch.long, device=env.device)
        goal_block = env.goal_block.value
        goal_positions = torch.nonzero(blocks[goal_block] == 1, as_tuple=True)

        # If there are no correct blocks, return
        if len(goal_positions[0]) == 0:
            return []

        # Calculate the closest correct block to the leader
        goal_dists = self.heuristic(start_pos.repeat(goal_positions[0].shape[0], 1), goal_positions.permute(1, 0))
        closest_goal_idx = torch.argmin(goal_dists)
        goal_x, goal_y = goal_positions[0][closest_goal_idx], goal_positions[1][closest_goal_idx]
        goal_pos = torch.tensor([goal_x, goal_y], dtype=torch.long, device=env.device)

        # Initialize the open and closed sets
        open_set = [(0 + self.heuristic(start_pos, goal_pos), start_pos, [])]
        heapq.heapify(open_set)
        closed_set = set()

        # Loop until the open set is empty or the goal is reached
        while open_set:
            current_cost, current_pos, current_path = heapq.heappop(open_set)

            # If the goal is reached, return the path
            if torch.equal(current_pos, goal_pos):
                return current_path

            closed_set.add(tuple(current_pos.tolist()))

            # Generate the neighbors
            neighbors = [
                (current_pos[0], current_pos[1] + 1),
                (current_pos[0], current_pos[1] - 1),
                (current_pos[0] + 1, current_pos[1]),
                (current_pos[0] - 1, current_pos[1]),
            ]

            for neighbor in neighbors:
                neighbor_pos = torch.tensor(neighbor, dtype=torch.long, device=env.device)
                # Check if the neighbor is within the boundaries and not a wall
                if env.leader.is_legal(neighbor_pos[0].item(), neighbor_pos[1].item()):
                    new_cost = current_cost - self.heuristic(current_pos, goal_pos) + 1 + self.heuristic(neighbor_pos, goal_pos)
                    new_path = current_path + [tuple(neighbor_pos.tolist())]

                    # Check if the neighbor is in the closed set or the open set
                    if tuple(neighbor_pos.tolist()) in closed_set:
                        continue
                    in_open_set = False
                    for idx, entry in enumerate(open_set):
                        if torch.equal(entry[1], neighbor_pos):
                            in_open_set = True
                            if entry[0] > new_cost:
                                open_set[idx] = (new_cost, neighbor_pos, new_path)
                            break

                    if not in_open_set:
                        heapq.heappush(open_set, (new_cost, neighbor_pos, new_path))

        # If no path is found, return an empty list
        return []
    
    def get_next_action(self, path):
        """
        Converts the path into a sequence of actions
        """
        actions = []
        for i in range(1, len(path)):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            if x2 > x1:
                actions.append(Moves.RIGHT.value)
            elif x2 < x1:
                actions.append(Moves.LEFT.value)
            elif y2 > y1:
                actions.append(Moves.UP.value)
            else:
                actions.append(Moves.DOWN.value)
                
        self.actions = actions
        return actions[0]
