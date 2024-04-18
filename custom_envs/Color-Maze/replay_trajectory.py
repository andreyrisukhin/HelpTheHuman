import numpy as np
from fire import Fire

from src.color_maze import ColorMaze


def replay_trajectory(trajectory: np.ndarray):
    assert trajectory.ndim == 4  # (step, channel, height, width)
    env = ColorMaze()
    env.reset()
    for step in range(trajectory.shape[0]):
        obs = trajectory[step]
        print(f"Step {step}:")
        env.set_state_to_observation(obs)
        env.render()
        input("Press any key to continue:")


def main(filepath: str):
    trajectory = np.load(filepath)
    replay_trajectory(trajectory)


if __name__ == '__main__':
    Fire(main)
