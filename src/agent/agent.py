import random

import numpy as np


class RandomAgent:
  def __init__(self) -> None:
    self.min_x = -5
    self.max_x = 5
    self.min_y = -5
    self.max_y = 5

  def choose_action(self) -> tuple[np.float64]:
    x = random.randint(self.min_x, self.max_x)
    y = random.randint(self.min_y, self.max_y)
    angle = np.random.random() * np.pi / 2

    return x, y, angle

  def update(state, action, reward, next_state) -> None:
    pass