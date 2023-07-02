import argparse

from agent.agent import RandomAgent
from environment.map import Map
from tqdm import tqdm


def train(env, agent, n_episodes, n_squares):
  for episodes in tqdm(range(n_episodes)):
    env.reset()
    state = env.state()
    placed_squares = 0

    while placed_squares < n_squares:
      action = agent.choose_action()
      next_state, reward, done = env.step(action)
      agent.update(state, action, reward, next_state)
      placed_squares += done
      state = next_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes')
    parser.add_argument('--squares')
    args = parser.parse_args()
    n_episodes = args.episodes
    n_squares = args.squares

    env = Map()
    agent = RandomAgent()

    train(n_episodes, n_squares)
