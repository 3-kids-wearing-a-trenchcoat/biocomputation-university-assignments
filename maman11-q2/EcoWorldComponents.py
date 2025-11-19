from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import copy

# constants
DEFAULT_SIZE = 500
INT8_MIN = -128
INT8_MAX = 127
UINT8_MAX = 255
SURFACE_STATES = 120
# component-specific constants
DEFAULT_LAND_PORTION = 0.3 # portion of the surface that is made up of (initially) dry land
DEFAULT_FOREST_COVERAGE = 0.4 # portion of dry land covered by forest
DEFAULT_FOREST_GERMINATION = 50 # maximum number of cells from germination center for forest
DEFAULT_INDUSTRY_GERM_LIMIT = 3
DEFAULT_INDUSTRY_QUANTITY = 700 # number of cells on dry land to be designated as industrial tiles
DEFAULT_POLLUTION_RATE = 2 # how much pollution each industry tile spawns at each iteration
DEFAULT_FOREST_CLEANING_RATE = 2 # how much pollution each forest tile removes at each iteration
START_POLLUTION = 0 # how much pollution should each cell start out with, mostly for debugging
WIND_STATES = 200 # maximum wind speed in any direction

class Surface:
    """Representing the world surface, each cell is described by surface elevation relative to some initial sea level"""

    def germinate_ground(self, x:np.int64, y:np.int64, height:int) -> int:
        if height <= 0:
            return 0
        if self.mat[x][y] != 0:
            if self.mat[x][y] < height:
                self.mat[x][y] += 1
            return 0
        self.mat[x][y] = height
        added_elevation = 1
        for new_x, new_y in ((x+1, y), (x-1, y), (x, y+1), (x,y-1)):
            new_x, new_y = new_x % self.mat.shape[0], new_y % self.mat.shape[1]
            added_elevation += self.germinate_ground(new_x, new_y, height - 1)
        return added_elevation

    def __init__(self, rnd_gen:np.random.Generator, size: int = DEFAULT_SIZE,
                 land_portion: float=DEFAULT_LAND_PORTION):
        """
        :param rnd_gen: a numpy.random.Generator object
        :param size: size of world edge (defaults to DEFAULT_SIZE)
        :param land_portion: fraction of the surface taken up by land (defaults to DEFAULT_LAND_PORTION)
        """
        self.mat = np.zeros((size, size), dtype=np.int8)
        target_land_coverage = np.square(size) * land_portion
        with tqdm(total=target_land_coverage, desc="Generating land elevation") as pbar:
            current_coverage = 0
            # while np.sum(np.where(self.mat > 0, 1, 0)) < target_land_coverage:
            while current_coverage < target_land_coverage:
                # randomly choose germination points on the world map
                x, y = rnd_gen.integers(size//8, 7*(size//8)), rnd_gen.integers(size // 3, 3*(size//4))
                added_coverage = self.germinate_ground(x, y, SURFACE_STATES)
                pbar.update(added_coverage)
                current_coverage += added_coverage




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
        return np.where(self.mat - self.surface.mat > 0, True, False).astype(np.bool)

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
    def __init__(self, rnd_gen:np.random.Generator, water:Water):
        """
        Create Wind cellular automata
        :param rnd_gen: numpy random number generator, should be shared by all components for consistency
        :param water: Water object
        """
        # initialize core variables
        self.water = water # water is used for the surface as elevation changes dictate wind erosion
                           # water can change elevation in this CA, surface remains static
        self.sn = np.zeros((water.mat.shape[0], water.mat.shape[1]), dtype=np.int64) # south-north wind
        self.ew = np.zeros_like(self.sn) # east-west wind
        self.vert = True # whether the wind in this iteration is vertical (sn) or horizontal (ew)
        self.alpha = 0.5 # diffusion rate of wind
        # pseudo-random wind spawning variables kind-of sort-of modeled after real world wind movement
        self.sixth = 0
        w = 2
        self.new_wind = {0: (w, -w),
                         1: (-w, w),
                         2: (w, -w),
                         3: (-w, -w),
                         4: (w, w),
                         5: (-w, -w)}

        self.sixth_length = self.sn.shape[0] // 6
        self.add_new_wind = True
        # propagation direction matrices, will make propagation calculations less costly
        self.move_n = np.zeros(self.water.mat.shape, dtype=np.bool)
        self.move_s = np.zeros(self.water.mat.shape, dtype=np.bool)
        self.move_e = np.zeros(self.water.mat.shape, dtype=np.bool)
        self.move_w = np.zeros(self.water.mat.shape, dtype=np.bool)

    def spawn_wind(self) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        output_sn = np.zeros_like(self.sn, dtype=np.int16)
        output_ew = np.zeros_like(self.ew, dtype=np.int16)
        # # find all cells where the cell and its neighbors are all equal to 0
        calm = np.logical_and(self.sn == 0, self.ew == 0)
        calm_area = (calm & np.roll(calm, 1, 0) & np.roll(calm, -1, 0) &
                     np.roll(calm, 1, 1) & np.roll(calm, -1, 1))
        spawn_mask = np.zeros(output_sn.shape, dtype=bool)
        spawn_start = self.sixth_length * self.sixth
        spawn_end = spawn_start + self.sixth_length
        spawn_mask[spawn_start : spawn_end] = True
        spawn_mask = spawn_mask & calm_area
        output_sn[spawn_mask] = self.new_wind[self.sixth][0]
        output_ew[spawn_mask] = self.new_wind[self.sixth][1]
        if self.add_new_wind:
            output_sn, output_ew = -output_sn, -output_ew
        if self.sixth == 5:
            self.sixth = 0
            self.add_new_wind = not self.add_new_wind
        else:
            self.sixth += 1
        return output_sn, output_ew

    def obstacle_matrix(self, shift:int, axis:int) -> NDArray:
        """
        Return a matrix representing how much of an obstacle the terrain is in a given direction
        :param shift:
        :param axis:
        :return: A matrix where each value represents how much wind erosion to apply to a wind source
        """
        return np.where(self.water.mat < np.roll(self.water.mat, shift, axis),
                        np.roll(self.water.mat, shift, axis) - self.water.mat, 0)

    def sum_with_wind_erosion(self) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        sn_sum, ew_sum = np.zeros_like(self.sn), np.zeros_like(self.ew)
        for shift, axis in [(1,0), (-1, 0), (1, 1), (-1, 1)]:
            block = self.obstacle_matrix(shift, axis)

            neighbor_abs_sn = np.absolute(np.roll(self.sn, shift, axis))
            neighbor_sign_sn = np.sign(np.roll(self.sn, shift, axis))
            neighbor_abs_ew = np.absolute(np.roll(self.ew, shift, axis))
            neighbor_sign_ew = np.sign(np.roll(self.ew, shift, axis))

            sn_sum += np.where(neighbor_abs_sn > block,
                               neighbor_sign_sn * (neighbor_abs_sn - block), 0)
            ew_sum += np.where(neighbor_abs_ew > block,
                               neighbor_sign_ew * (neighbor_abs_ew - block), 0)
        return sn_sum, ew_sum

    def wind_interaction(self) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        # calculate terrain blocks
        # south_block = self.obstacle_matrix(1, 0)
        # north_block = self.obstacle_matrix(-1, 0)
        # east_block = self.obstacle_matrix(1, 1)
        # west_block = self.obstacle_matrix(-1, 1)
        # sum of neighborhood wind strength, for sn and ew
        # applying erosion
        # sn_sum = (np.roll(self.sn, 1, 0) - south_block +
        #           np.roll(self.sn, -1, 0) + north_block +
        #           np.roll(self.sn, 1, 1) - east_block +
        #           np.roll(self.sn, -1, 1) + west_block)
        # ew_sum = (np.roll(self.ew, 1, 0) - south_block +
        #           np.roll(self.ew, -1, 0) + north_block +
        #           np.roll(self.ew, 1, 1) - east_block +
        #           np.roll(self.ew, -1, 1) + west_block)
        sn_sum, ew_sum = self.sum_with_wind_erosion()
        # # clip values to WIND_STATES
        # sn_sum, ew_sum = np.clip(sn_sum, -WIND_STATES, WIND_STATES), np.clip(ew_sum, -WIND_STATES, WIND_STATES)
        # get averages
        sn_avg, ew_avg = sn_sum // 4, ew_sum // 4
        # apply wind diffusion and return
        output_sn = self.sn + (self.alpha * np.rint(sn_avg).astype(np.int64))
        output_ew = self.ew + (self.alpha * np.rint(ew_avg).astype(np.int64))
        # clip values to WIND_STATES
        # output_sn, output_ew = np.clip(output_sn, -WIND_STATES, WIND_STATES), np.clip(output_ew, -WIND_STATES, WIND_STATES)
        return output_sn.astype(np.int16), output_ew.astype(np.int16)

    def update_wind(self) -> None:
        # calculate new values
        spawned_sn, spawned_ew = self.spawn_wind()
        inter_sn, inter_ew = self.wind_interaction()
        self.sn, self.ew = spawned_sn + inter_sn, spawned_ew + inter_ew
        # update propagation direction matrices
        self.move_s = np.where(self.sn > 0, True, False)
        self.move_n = np.where(self.sn < 0, True, False)
        self.move_e = np.where(self.ew > 0, True, False)
        self.move_w = np.where(self.ew < 0, True, False)

    def wind_propagation(self, mat: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """propagate the given matrix (representing pollution or clouds, for example) according to the wind"""
        smat = mat.astype(np.int16)
        s_movers = np.where(self.move_s, -smat, 0)
        s_movers -= np.roll(s_movers, 1, 0)
        n_movers = np.where(self.move_n, -smat, 0)
        n_movers -= np.roll(n_movers, -1, 0)
        e_movers = np.where(self.move_e, -smat, 0)
        e_movers -= np.roll(e_movers, 1, 1)
        w_movers = np.where(self.move_w, -smat, 0)
        w_movers -= np.roll(w_movers, -1, 1)
        # output = smat + s_movers + n_movers + e_movers + w_movers
        # return output.astype(np.uint8)
        if self.vert:
            return (smat + s_movers + n_movers).astype(np.uint8)
        return (smat + e_movers + w_movers).astype(np.uint8)

    def copy(self) -> Wind:
        output = copy.copy(self)  # shallow copy, only mat needs to be deep-copied, the rest are pointers
        # to components that will handle their own update
        output.sn = self.sn.copy()
        output.ew = self.ew.copy()
        output.vert = not self.vert
        return output

    def __iter__(self):
        return self

    def __next__(self):
        output = self.copy()
        output.update_wind()
        return output

    def update_components(self, updated_water: Water) -> None:
        self.water = updated_water


class Forest:
    """Representing forest tiles which clean pollution"""
    def germinate_forest(self, x:int, y:int, limit:int, land_map:NDArray[np.bool]) -> int:
        if limit <= 0 or not land_map[x][y]:
            return 0
        self.mat[x][y] = True
        # self.land_coords.remove((x,y))
        new_forest_tiles = 1
        for new_x, new_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            new_x, new_y = new_x % self.mat.shape[0], new_y % self.mat.shape[1]
            if self.mat[new_x][new_y]:
                continue
            new_forest_tiles += self.germinate_forest(new_x, new_y, limit - 1, land_map)
        return new_forest_tiles

    def __init__(self, rnd_gen:np.random.Generator, surface:Surface, water:Water,
                 cover:float=DEFAULT_FOREST_COVERAGE, germ_limit:int=DEFAULT_FOREST_GERMINATION):
        self.water = water
        self.mat = np.zeros(surface.mat.shape, dtype=np.bool)
        land_map = np.where(water.mat - surface.mat <= 0, True, False).astype(np.bool)
        target_coverage = cover * np.sum(land_map)
        land_x, land_y = np.nonzero(land_map)
        self.land_coords = list(zip(land_x, land_y))
        with tqdm(total=target_coverage, desc="Generating forest tiles") as pbar:
            current_coverage = 0
            while current_coverage < target_coverage:
                x, y = rnd_gen.choice(self.land_coords)
                new_tiles = self.germinate_forest(x, y, germ_limit, land_map)
                pbar.update(new_tiles)
                current_coverage += new_tiles

        self.land_coords = None

    def __iter__(self):
        return self

    def __next__(self):
        output = copy.copy(self)
        overlap = self.mat & self.water.get_water_position()
        if np.any(overlap):
            output.mat = np.where(overlap, False, output.mat).astype(np.bool)
        return output

    def update_components(self, water:Water) -> None:
        self.water = water

class Industry:
    """Representing industrial tiles which spew pollution"""
    def germinate_industry(self, x:int, y:int, limit:int,
                           land_map:NDArray[np.bool], forest_map:NDArray[np.bool]) -> int:
        if limit <= 0 or not land_map[x][y]:
            return 0
        self.mat[x][y] = True
        # self.land_coords.remove((x,y))
        new_industry_tiles = 1
        for new_x, new_y in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            new_x, new_y = new_x % self.mat.shape[0], new_y % self.mat.shape[1]
            if self.mat[new_x][new_y] or forest_map[new_x][new_y]:
                continue
            new_industry_tiles += self.germinate_industry(new_x, new_y, limit - 1, land_map, forest_map)
        return new_industry_tiles

    def __init__(self, rnd_gen:np.random.Generator, surface:Surface, water:Water, forest:Forest,
                 num:int=DEFAULT_INDUSTRY_QUANTITY, germ_limit:int=DEFAULT_INDUSTRY_GERM_LIMIT):
        self.water = water
        self.mat = np.zeros(surface.mat.shape, dtype=np.bool)
        land_map = np.where(water.mat - surface.mat <= 0, True, False).astype(np.bool)
        free_space = (~forest.mat) & land_map
        free_x, free_y = np.nonzero(free_space)
        self.free_coords = list(zip(free_x, free_y))
        with tqdm(total=num, desc="Placing industrial tiles") as pbar:
            total = 0
            while total < num:
                x, y = rnd_gen.choice(self.free_coords)
                # self.mat[x][y] = True
                # self.free_coords.remove((x,y))
                # total += 1
                # pbar.update(1)
                new_tiles = self.germinate_industry(x, y, germ_limit, land_map, forest.mat)
                pbar.update(new_tiles)
                total += new_tiles
        self.free_coords = None

    def __iter__(self):
        return self

    def __next__(self):
        output = copy.copy(self)
        overlap = self.mat & self.water.get_water_position()
        if np.any(overlap):
            output.mat = np.where(overlap, False, output.mat).astype(np.bool)
        return output

    def update_components(self, water: Water) -> None:
        self.water = water


class Pollution:
    """Represents pollution, which is generated by industrial tiles and propagates by the wind"""
    def __init__(self, industry:Industry, forest:Forest, wind:Wind, spawn_rate:int = DEFAULT_POLLUTION_RATE,
                 cleaning_rate:int = DEFAULT_FOREST_CLEANING_RATE):
        self.industry = industry
        self.forest = forest
        self.wind = wind
        self.spawn_rate = spawn_rate
        self.cleaning_rate = cleaning_rate
        self.mat = np.zeros(industry.mat.shape, dtype=np.uint8) + START_POLLUTION

    def clear_and_spawn(self) -> None:
        delta = (self.spawn_rate * self.industry.mat.astype(np.int16) -
                 self.cleaning_rate * self.forest.mat.astype(np.int16))
        new_mat = self.mat.astype(np.int16) + delta
        new_mat = np.clip(new_mat, 0, UINT8_MAX)
        self.mat = new_mat.astype(np.uint8)

    def propagate_full_cells(self):
        """have pollution spread to nearby cells if the cell is full"""
        full = np.where(self.mat == UINT8_MAX, True, False).astype(np.bool)
        change = np.zeros(self.mat.shape, dtype=np.int16)
        change[full] = UINT8_MAX // 5
        change += (np.roll(change, 1, 0) + np.roll(change, -1, 0) +
                   np.roll(change, 1, 1) + np.roll(change, 1, -1))
        change[full] = - (UINT8_MAX - change[full])
        # self.mat += change
        output = self.mat.astype(np.int16) + change
        self.mat = output.astype(np.uint8)

    def __iter__(self):
        return self

    def __next__(self):
        output = copy.copy(self)
        output.mat = self.mat.copy()
        output.clear_and_spawn()
        # output.wind_propagation()
        output.mat = self.wind.wind_propagation(output.mat.astype(np.uint8))
        output.propagate_full_cells()
        return output

    def get_total_pollution(self) -> int:
        return np.sum(self.mat)

    def update_components(self, industry:Industry, forest:Forest, wind:Wind) -> None:
        self.industry = industry
        self.forest = forest
        self.wind = wind

# class Temperature:
#     """Representing temperature and the way it changes due to the sun, greenhouse effects and albedo"""
#     def __init__(self, water:Water, wind:Wind, pollution:Pollution):
#         # TODO: add glaciers to Temperature
#         self.water = water
#         self.wind = wind
#         self.pollution = pollution
#         self.temp = np.zeros(water.mat.shape, dtype=np.int16)
#