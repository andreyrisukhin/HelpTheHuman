import numpy as np
from fire import Fire

from color_maze import ColorMaze, IDs


def replay_trajectory(trajectory: np.ndarray, goal_info: np.ndarray | None = None):
    assert trajectory.ndim == 5  # (minibatch_step, history, channel, height, width) 
    assert goal_info.ndim == 3 # (minibatch step, history, goal_dim = 3)

    most_recent_trajectory = trajectory[:,-1]
    most_recent_goal_info = goal_info[:,-1]
    breakpoint()
    env = ColorMaze()
    env.reset()
    for step in range(most_recent_trajectory.shape[0]):
        obs = most_recent_trajectory[step]
        print(f"Step {step}:")
        # breakpoint()
        env.set_state_to_observation(obs)
        env.render()
        if most_recent_goal_info is not None:
            goal_idx = np.argmax(most_recent_goal_info[step])
            print(f'Current goal: {IDs(goal_idx)}')
        input("Press any key to continue:")


def main(trajectory_filepath: str, goal_info_filepath: str):
    trajectory = np.load(trajectory_filepath)
    goal_info = np.load(goal_info_filepath)
    replay_trajectory(trajectory, goal_info)


if __name__ == '__main__':
    Fire(main)
