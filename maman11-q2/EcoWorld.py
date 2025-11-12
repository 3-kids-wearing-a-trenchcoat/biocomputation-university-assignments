from __future__ import annotations
from numba import njit
import numpy as np
from numpy.typing import NDArray
import tqdm

# constants
DEFAULT_SIZE = 500
DEFAULT_LAND_PORTION = 0.4
DEFAULT_LAND_ASPECT_RATIO = 2
INT8_MIN = -128, INT8_MAX = 127

class Surface:
    """Representing the world surface, each cell is described by surface elevation relative to some initial sea level"""

    def __init__(self, rnd_gen:np.random.Generator, size: int = DEFAULT_SIZE,
                 land_portion: float=DEFAULT_LAND_PORTION, land_aspect_ratio:float = DEFAULT_LAND_ASPECT_RATIO):
        """

        :param rnd_gen: a numpy.random.Generator object
        :param size: surface size (defaults to DEFAULT_SIZE)
        :param land_portion: fraction of the surface taken up by land (defaults to DEFAULT_LAND_PORTION)
        :param land_aspect_ratio: ratio of land width to length height (defaults to DEFAULT_LAND_ASPECT_RATIO)
        """
        # self.mat = np.zeros((size, size), dtype=np.int8)
        self.mat = np.ones((size, size)) * -1
        land_quantity = np.floor(size * size * land_portion)
        land_height = np.floor(land_quantity * land_portion / land_aspect_ratio)
        land_width = land_height * land_aspect_ratio
        mid_point = size // 2
        land_top, land_bottom = mid_point - (land_height // 2), mid_point + (land_height // 2) + 1
        land_left, land_right = mid_point - (land_width // 2), mid_point + (land_width // 2) + 1
        land = self.mat[land_top:land_bottom, land_left:land_right]
        land = np.ones(land.shape)

    def get_height_mask(self, below:int=INT8_MAX, above:int=INT8_MIN) -> NDArray[np.bool]:
        """Return a mask where a cell is true if it is lower than 'below' (inclusive) and higher than 'above' (exclusive)"""
        return (self.mat <= below) & (self.mat > above)


class Water:
    """Represents water on the world's surface.
    Water in a tile moves to a nearby tile iff it is without water and is lower than the current tile"""

    def __init__(self, surface:Surface, initial_water_level:int=0):
        self.surface = surface
        self.has_water = self.surface.get_height_mask(initial_water_level)
        self.height_with_water = surface.mat.copy()
        self.height_with_water[self.has_water] = 0

    def directional_equalize(self, i:int, axis:int) -> None:
        """apply flow in the direction (i,axis),
        meaning water will flow one unit at a time from (x,y) to (x+i,y+axis)"""
        # set to -1 in all cells whose equivalent has water and is higher than the cell -i of it
        change = (self.has_water &
                  (self.height_with_water > np.roll(self.height_with_water, -i, axis))).astype(np.int8) * -1
        # set 1 in all cells where the cell to the i of it (top/bottom/left/right) was set to -1
        change += np.roll(change, i, axis) * -1
        # update changes
        self.height_with_water += change
        self.has_water = self.height_with_water > self.surface.mat


    def equalize(self):
        """Take a step towards equalizing all water tiles
        Water moves from one cell to another if the other cell's height (including water) is lower than this one's"""
        [self.directional_equalize(i, j) for i, j in ((1,0), (-1,0), (1,1), (-1,1))]