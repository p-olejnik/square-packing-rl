from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon


@dataclass
class Position:
    x: np.float64
    y: np.float64


@dataclass
class Square:
    """
    Class that represents square placed on the map.
    Square's position is represented by coordinates of one of
    its corners and rotation - in radians from range [0, pi/2).
    We assume unit length of its side.
    """

    pos: Position
    angle: np.float64

    def get_corners(self) -> list[tuple[np.float64]]:
        corner0 = (self.pos.x, self.pos.y)
        corner1 = (self.pos.x + np.sin(self.angle), self.pos.y + np.cos(self.angle))
        corner2 = (
            self.pos.x + np.cos(self.angle + np.pi / 4) * np.sqrt(2),
            self.pos.y + np.sin(self.angle + np.pi / 4) * np.sqrt(2),
        )
        corner3 = (
            self.pos.x + np.cos(self.angle + np.pi / 2) * np.sqrt(2),
            self.pos.y + np.sin(self.angle + np.pi / 2) * np.sqrt(2),
        )
        return [corner0, corner1, corner2, corner3]

    def representation(self) -> NDArray[np.float64]:
        return np.array([self.pos.x, self.pos.y, self.angle])

    @staticmethod
    def squares_overlap(square1, square2) -> bool:
        square1_poly = Polygon(square1.get_corners())
        square2_poly = Polygon(square2.get_corners())
        return square1_poly.intersects(square2_poly)


@dataclass
class BoundingSquare:
    min_x: np.float64
    max_x: np.float64
    min_y: np.float64
    max_y: np.float64

    def side_len(self) -> np.float64:
        return np.max(np.abs(self.max_x - self.min_x), np.abs(self.max_y - self.min_y))

    def corners(self) -> list[tuple[np.float64]]:
        side_len = self.side_len()
        bounding_square_corners = [
            (self.min_x, self.min_y),
            (self.min_x + side_len, self.min_y),
            (self.min_x, self.min_y + side_len),
            (self.min_x + side_len, self.min_y),
        ]
        return bounding_square_corners


class Map:
    """Class that represents environment where squares are placed."""

    def __init__(self) -> None:
        self.squares: list[Square] = []

    def _can_place_square(self, new_square: Square) -> bool:
        for square in self.squares:
            if square.has_overlap(new_square):
                return False

        return True

    def step(self, action: tuple[np.float64]) -> tuple:
      x, y, angle = action
      done = self.place_square(x, y, angle)
      reward = -1 + 2 * done
      state = self.state()
      return state, reward, done

    def place_square(self, x: np.float64, y: np.float64, angle: np.float64) -> bool:
        assert angle <= np.pi / 2
        new_square = Square(Position(x, y), angle)

        if not self._can_place_square(new_square):
            return False

        self.squares.append(new_square)
        return True

    def minimal_bounding_square(self) -> list[np.float64]:
        """Return corners of smallest square that contains all quares on the map"""
        if not self.squares:
            return [0, 0, 0, 0]

        init_corners = self.squares[0]
        min_x = np.min([init_corners[i][0] for i in range(4)])
        max_x = np.max([init_corners[i][0] for i in range(4)])
        min_y = np.min([init_corners[i][1] for i in range(4)])
        max_y = np.max([init_corners[i][1] for i in range(4)])

        for square in self.squares[1:]:
            for corner in square.get_corners():
                min_x = np.min([min_x, corner[0]])
                max_x = np.max([max_x, corner[0]])
                min_y = np.min([min_y, corner[1]])
                max_y = np.max([max_y, corner[1]])

        return BoundingSquare(min_x, max_x, min_y, max_y)

    def state(self) -> NDArray[np.float64]:
        """Return triples representing a square for each square on the map"""
        return np.array([square.representation() for square in self.squares])

    def packing_density(self):
        squares_area = len(self.squares)  # assuming unit squares
        bounding_square_area = self.minimal_bounding_square().side_len() ** 2
        packing_density = np.divide(squares_area, bounding_square_area)
        return packing_density

    def visualize(self):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        for square in self.squares:
            square_corners = square.get_corners()
            rectangle = Rectangle(
                square_corners[0], 1, 1, angle=-np.rad2deg(square.angle), color="blue"
            )
            ax.add_patch(rectangle)

        bounding_square = self.minimal_bounding_square()

        plt.xlim(bounding_square.min_x - 1, bounding_square.max_x + 1)
        plt.ylim(bounding_square.min_y - 1, bounding_square.max_y + 1)
        plt.grid(True)
        plt.show()

    def reset(self) -> None:
      self.squares.clear()
