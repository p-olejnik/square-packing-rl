from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon


@dataclass
class Position:
  x: np.float64
  y: np.float64


@dataclass 
class Square:
  """
  Class that represents square placed on the map. Square's position is represented by coordinates of one of
  its corners and rotation - in radians from range [0, pi/2). We assume unit length of its side.
  """
  pos: Position 
  angle: np.float64

  def get_corners(self) -> list[tuple[np.float64]]:
    corner0 = (
      self.pos.x, 
      self.pos.y
    )
    corner1 = (
      self.pos.x + np.sin(self.angle),
      self.pos.y + np.cos(self.angle)
    )
    corner2 = (
      self.pos.x + np.cos(self.angle + np.pi / 4) * np.sqrt(2), 
      self.pos.y + np.sin(self.angle + np.pi / 4) * np.sqrt(2)
    )
    corner3 = (
      self.pos.x + np.cos(self.angle + np.pi / 2) * np.sqrt(2), 
      self.pos.y + np.sin(self.angle + np.pi / 2) * np.sqrt(2)
    )
    return [corner0, corner1, corner2, corner3]
  
  @staticmethod
  def squares_overlap(square1, square2) -> bool:
    square1_poly = Polygon(square1.get_corners())
    square2_poly = Polygon(square2.get_corners())
    return square1.intersects(square2)


class Map:
  """Class that represents environment where squares are placed."""
  def __init__(self) -> None:
    self.squares: list[Square] = []

  def _can_place_square(self, new_square: Square) -> bool:
    for square in self.squares:
      if square.has_overlap(new_square):
        return False
    
    return True

  def place_square(self, x: np.float64, y: np.float64, rotation: np.float64) -> bool:
    assert rotation <= np.pi / 2
    new_square = Square(Position(x, y), rotation)
    
    if not self._can_place_square(new_square):
      return False
    
    self.squares.append(new_square)
    return True
  
  def visualize(self):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for square in self.squares:
      square_corners = square.get_corners()
      rectangle = Rectangle(square_corners[0], 1, 1, angle=-np.rad2deg(square.angle), color='blue')
      ax.add_patch(rectangle)

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True)
    plt.show()

