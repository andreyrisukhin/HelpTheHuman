import heapq
import numpy as np
from color_maze import Moves, IDs
import torch


def heuristic(agent_pos, goal_pos):
        """
        Manhattan distance as the heuristic function
        """
        return torch.sum(torch.abs(agent_pos - goal_pos), dim=-1)
        
def a_star_search(env, agent):
    """
    Performs A* search to find the shortest path for the leader to the closest correct block
    """
    blocks = env.blocks
    start_pos = torch.tensor([agent.x, agent.y], device=env.device)
    # print(start_pos)
    goal_block = env.goal_block.value
    goal_positions = torch.nonzero(blocks[goal_block] == 1, as_tuple=True)
    # breakpoint()
    
    # Inv: There should always be correct blocks
    if len(goal_positions[0]) == 0:
        assert False, "No correct blocks found"

    # Calculate the closest correct block to the leader
    goal_positions_tensor = torch.stack(goal_positions, dim=1)
    goal_dists = heuristic(start_pos.repeat(goal_positions_tensor.shape[0], 1), goal_positions_tensor)
    closest_goal_idx = torch.argmin(goal_dists)
    goal_pos = goal_positions_tensor[closest_goal_idx]
    # convert goal_pos to tuple
    # target = tuple(goal_pos.tolist())

    # Initialize the open and closed sets
    open_set = [(0 + heuristic(start_pos, goal_pos).item(), start_pos.tolist(), [])]
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
            neighbor_pos = torch.tensor(neighbor, dtype=torch.long, device=env.device)
            # Check if the neighbor is within the boundaries and not a wall
            if agent.is_legal(neighbor_pos[0].item(), neighbor_pos[1].item()):
                
                # if neighbor_pos isn't a goal block, penalize
                penalize = False
                for penalty_color in set((IDs.BLUE, IDs.RED, IDs.GREEN)) - set((env.goal_block, )):
                    if blocks[penalty_color.value][neighbor_pos[0], neighbor_pos[1]] == 1:
                        new_cost = 999999
                        penalize = True
                    
                if not penalize:
                    new_cost = current_cost - heuristic(torch.tensor(current_pos, device=env.device), goal_pos).item() + 1 + heuristic(neighbor_pos, goal_pos).item()
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

class AStarLeader:
    "Uses A* search to find the shortest path to the closest correct block for the leader."
    def __init__(self):
        self.goal_block_color = None

    def __call__(self, env, agent):
        
        # only the leader can use this policy
        assert (
            agent == env.agents[0]
        ), "AStarLeader can only be applied to leader."
        
        # if the goal color just switched re-search or agent just collected a block
        # TODO: Check whenever a block spawned (follower might have collected a block)
        # if self.goal_block_color != self.env.goal_block or (self.env.leader.x == self.target[0] and self.env.leader.y == self.target[1]):
        #     self.path = self.a_star_search()
        #     print(self.path)
        #     self.goal_block_color = self.env.goal_block
        #     assert self.path, "No path found"
        # else:
        #     self.path = self.path[1:]
            
        path = a_star_search(env, agent)
        # print(self.path)

        x1, y1 = agent.x, agent.y
        x2, y2 = path[0]
        if x1 < x2:
            return Moves.RIGHT.value
        elif x1 > x2:
            return Moves.LEFT.value
        elif y1 < y2:
            return Moves.UP.value
        else:
            return Moves.DOWN.value
    
class CopyFollower:
    "Copies AStarLeader policy but doesn't know the goal block until after the leader moves."
    def __init__(self):
        self.goal_block_color = None

    def __call__(self, env, agent):
        agent = self.env.agents[self.agent_id]
        # only the leader can use this policy
        assert (
            agent == env.agents[1]
        ), "CopyFollower can only be applied to leader."
            
        path = a_star_search(env, agent)

        x1, y1 = agent.x, agent.y
        x2, y2 = path[0]
        if x1 < x2:
            return Moves.RIGHT.value
        elif x1 > x2:
            return Moves.LEFT.value
        elif y1 < y2:
            return Moves.UP.value
        else:
            return Moves.DOWN.value
