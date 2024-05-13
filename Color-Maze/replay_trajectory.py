import numpy as np
from fire import Fire

from color_maze import ColorMaze, IDs


def replay_trajectory(trajectory: np.ndarray, goal_info: np.ndarray | None = None):
    assert trajectory.ndim == 4  # (minibatch_step, channel, height, width) 
    assert goal_info is None or goal_info.ndim == 2 # (minibatch step, goal_dim = 3)

    env = ColorMaze()
    env.reset()
    for step in range(trajectory.shape[0]):
        obs = trajectory[step]
        print(f"Step {step}:")
        env.set_state_to_observation(obs)
        env.render()
        if goal_info is not None:
            goal_idx = np.argmax(goal_info[step])
            print(f'Current goal: {IDs(goal_idx)}')
        input("Press any key to continue:")


def main(trajectory_filepath: str, goal_info_filepath: str | None = None):
    trajectory = np.load(trajectory_filepath)
    if goal_info_filepath is None:
        goal_info_filepath = trajectory_filepath.replace('trajectory_', 'goal_info_')
    goal_info = np.load(goal_info_filepath)
    replay_trajectory(trajectory, goal_info)


if __name__ == '__main__':
    Fire(main)
