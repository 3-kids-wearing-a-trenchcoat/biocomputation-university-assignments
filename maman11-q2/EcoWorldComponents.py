from __future__ import annotations
from numba import njit
import numpy as np
from numpy.typing import NDArray
import tqdm
import copy

# constants
DEFAULT_SIZE = 500
DEFAULT_LAND_PORTION = 0.3
INT8_MIN = -128
INT8_MAX = 127
SURFACE_STATES = 20

class Surface:
    """Representing the world surface, each cell is described by surface elevation relative to some initial sea level"""

    # @njit()
    def germinate_ground(self, x:np.int64, y:np.int64, height:int) -> None:
        if height <= 0:
            return
        if self.mat[x][y] != 0:
            if self.mat[x][y] < height:
                self.mat[x][y] += 1
            return
        self.mat[x][y] = height
        for new_x, new_y in ((x+1, y), (x-1, y), (x, y+1), (x,y-1)):
            if new_x >= self.mat.shape[0]:
                new_x -= self.mat.shape[0]
            elif new_x < 0:
                new_x += self.mat.shape[0] - 1
            if new_y >= self.mat.shape[1]:
                new_y -= self.mat.shape[1]
            elif new_y < 0:
                new_y += self.mat.shape[1] - 1
            self.germinate_ground(new_x, new_y, height - 1)

    def __init__(self, rnd_gen:np.random.Generator, size: int = DEFAULT_SIZE,
                 land_portion: float=DEFAULT_LAND_PORTION):
        """
        :param rnd_gen: a numpy.random.Generator object
        :param size: size of world edge (defaults to DEFAULT_SIZE)
        :param land_portion: fraction of the surface taken up by land (defaults to DEFAULT_LAND_PORTION)
        """
        self.mat = np.zeros((size, size), dtype=np.int8)
        target_land_coverage = np.square(size) * land_portion
        while np.sum(np.where(self.mat > 0, 1, 0)) < target_land_coverage:
            # randomly choose germination points on the world map
            x, y = rnd_gen.integers(size//8, 7*(size//8)), rnd_gen.integers(size // 3, 3*(size//4))
            self.germinate_ground(x, y, SURFACE_STATES)




class Water:
    """Represents water on the world's surface.
    Water in a tile moves to a nearby tile iff it is without water and is lower than the current tile"""

    def __init__(self, surface:Surface, initial_water_level:int=1):
        self.surface = surface
        self.mat = np.where(surface.mat < initial_water_level, initial_water_level, surface.mat)


    def directional_equalize(self, shift:int, axis:int) -> None:
        """apply flow in the direction of shift on the given axis (shift and axis as used in np.roll)"""
        # set -1 to cells from which water will move
        # change = np.where(self.mat > np.roll(self.mat, -shift, axis) and (self.mat - self.surface.mat > 0), -1, 0)
        change = np.where(np.logical_and(self.mat > np.roll(self.mat, -shift, axis),
                                         self.mat - self.surface.mat > 0), -1, 0).astype(np.int8)

        # set 1 to cells into which water will move
        change += np.roll(change, shift, axis) * -1
        self.mat += change
        self.mat = np.clip(self.mat, 0, SURFACE_STATES)

    def equalize(self):
        """Take a step towards equalizing all water tiles
        Water moves from one cell to another if the other cell's height (including water) is lower than this one's"""
        [self.directional_equalize(i, j) for i, j in ((1,0), (-1,0), (1,1), (-1,1))]

    def __iter__(self) -> Water:
        return self

    def __next__(self) -> Water:
        # self.equalize()
        # return self
        output = self.copy()
        output.equalize()
        return output

    def copy(self) -> Water:
        output = copy.copy(self)
        output.mat = self.mat.copy()
        return output

    def get_water_position(self) -> np.typing.NDArray[np.bool]:
        return np.where(self.mat - self.surface.mat > 0, True, False)

    def average_sea_level(self) -> float:
        water_pos = self.get_water_position()
        return self.mat[water_pos].mean()


class Wind:
    """
    Representation of wind.
    Each cell is made up of 2 uint8 values, the first for horizontal wind (positive towards East)
    and the second for vertical wind (positive towards South).
    Each cell's wind is influenced by its neighbors, making it so the strongest direction in the neighborhood
    is the cell's next wind direction, whereas its strength is what remains after subtracting the strength of other
    winds in the neighborhood.
    The central strip of the world will also generate wind if all neighbors are calm, to avoid a stable,
    completely calm state and to simulate wind resulting in the world's rotation at the (roughly) equator
    """
    def __init__(self, water:Water):
        self.water = water # water is used for the surface as elevation changes dictate wind erosion
                           # water can change elevation in this CA, surface remains static
        self.mat = np.zeros((water.mat.shape[0], water.mat.shape[1], 2), dtype=np.int8)
        self.wind_generation_range = (1 * (water.mat.shape[1] // 3), 2 * (water.mat.shape[1] // 3))

    def update_wind(self) -> None:
        wind_sum = (self.mat + np.roll(self.mat, 1, 0) + np.roll(self.mat, -1, 0) +
                                np.roll(self.mat, 1, 1) + np.roll(self.mat, -1, 1))
        # TODO: implement wind erosion based on terrain
        south_north, east_west = wind_sum[..., 0], wind_sum[..., 1]
        is_sn = np.absolute(south_north) > np.absolute(east_west)
        self.mat[..., 0] = np.where(is_sn,
                                    np.sign(south_north) * (np.absolute(south_north) - np.absolute(east_west)), 0)
        self.mat[..., 1] = np.where(is_sn,
                                    0, np.sign(east_west) * (np.absolute(east_west) - np.absolute(south_north)))

    def get_wind_direction(self) -> NDArray:
        # sn = np.where(self.mat[...,0] != 0, np.sign(self.mat[...,0]), 0)
        # ew = np.where(self.mat[...,1] != 0, np.sign(self.mat[...,1]), 0)
        # return sn + ew
        return np.where(self.mat[...,0] != 0, [np.sign(self.mat[...,0]), 0],
                        np.where(self.mat[...,1] != 0, [0, np.sign(self.mat[...,1])],
                                 0))

    def copy(self) -> Wind:
        output = copy.copy(self) # shallow copy, only mat needs to be deep-copied, the rest are pointers
                                 # to components that will handle their own update
        output.mat = self.mat.copy()
        return output

    def __iter__(self):
        return self

    def __next__(self):
        output = self.copy()
        output.update_wind()
        return output

    def update_components(self, updated_water:Water):
        self.water = updated_water