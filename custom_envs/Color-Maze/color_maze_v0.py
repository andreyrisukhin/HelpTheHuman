from pettingzoo.test import parallel_api_test

from env.color_maze import ColorMaze


if __name__ == "__main__":
    env = ColorMaze()
    parallel_api_test(env, num_cycles=1_000_000)
