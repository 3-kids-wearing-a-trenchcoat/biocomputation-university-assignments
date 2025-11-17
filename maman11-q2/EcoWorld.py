from concurrent.futures import ThreadPoolExecutor

import numpy as np

from EcoWorldComponents import *
import atexit

# constants
INITIAL_HISTORY_SIZE = 100
# colors
LIGHT_BLUE = (0, 150, 255)
BLACK = (0, 0, 0)
BROWN = (101, 67, 33)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
# PURPLE = (191, 64, 191)
OVERLAY_COLOR_ALPHA = 0.4

class EcoWorld:
    """Aggregation of all EcoWorldComponent classes into a single iterable class"""
    def __init__(self, rnd_gen:np.random.Generator, size:int = DEFAULT_SIZE):
        # Initialize components
        self.surface = Surface(rnd_gen, size)
        self.water = Water(self.surface)
        self.wind = Wind(rnd_gen, self.water)
        self.forest = Forest(rnd_gen, self.surface, self.water)
        self.industry = Industry(rnd_gen, self.surface, self.water, self.forest)
        self.pollution = Pollution(self.industry, self.forest, self.wind)
        # Data trackers
        # average sea level
        self.sea_level = np.zeros(INITIAL_HISTORY_SIZE, dtype=np.uint8)
        self.sea_level_ptr = 0
        # total pollution
        self.pollution_tracker = np.zeros(INITIAL_HISTORY_SIZE, dtype=np.uint64)
        self.pollution_tracker_ptr = 0
        # initialize toggles
        self.show_pollution_toggle = True
        # initialize parallelism
        self.executor = ThreadPoolExecutor(max_workers=4)
        atexit.register(self.executor.shutdown, wait=True)

    @staticmethod
    def double_array_size(a:np.typing.NDArray) -> np.typing.NDArray:
        output = np.empty(a.shape[0] * 2)
        output[:a.shape[0]] = a
        return output

    def update_trackers(self) -> None:
        # sea level
        self.sea_level[self.sea_level_ptr] = self.water.average_sea_level()
        self.sea_level_ptr += 1
        if self.sea_level_ptr == self.sea_level.shape[0]:
            self.sea_level = self.double_array_size(self.sea_level)
        # pollution
        self.pollution_tracker[self.pollution_tracker_ptr] = self.pollution.get_total_pollution()
        self.pollution_tracker_ptr += 1
        if self.pollution_tracker_ptr == self.pollution_tracker.shape[0]:
            self.pollution_tracker = self.double_array_size(self.pollution_tracker)

    def update_component_pointers(self):
        self.wind.update_components(self.water)
        self.forest.update_components(self.water)
        self.industry.update_components(self.water)
        self.pollution.update_components(self.industry, self.forest, self.wind)

    def __iter__(self):
        return self

    def __next__(self):
        comp_names = ["water", "wind", "forest", "industry", "pollution"]
        futures = {name: self.executor.submit(next, getattr(self, name))
                   for name in comp_names}
        for name, fut in futures.items():
            setattr(self, name, fut.result())
        # advance components by one step
        # self.water = next(self.water)
        # self.wind = next(self.wind)
        # self.forest = next(self.forest)
        # self.industry = next(self.industry)

        # update points of components to components
        self.update_component_pointers()
        # update trackers
        self.update_trackers()
        return self

    def get_water_grid(self) -> NDArray[np.uint8]:
        """
        Get a colored matrix representing the presence of water.
        Blue tiles have water in them, empty (0,0,0,0) tiles do not
        :return: 4D np.uint8 matrix representing RGBA values
        """
        pos = self.water.get_water_position()
        h, w = pos.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[pos] = (0, 150, 255, 255)
        rgba[~pos] = (0, 0, 0, 0)
        return rgba

    @staticmethod
    def overlay_color(origin:NDArray[np.uint8], mask:NDArray, color:tuple[int, int, int],
                      proportional:bool=True) -> NDArray:
        if not proportional:
            return np.where(mask,
                            ((1 - OVERLAY_COLOR_ALPHA)*origin) + OVERLAY_COLOR_ALPHA * np.full_like(origin, color),
                            origin)
        fg_a = mask.astype(np.float64) / UINT8_MAX
        bg_a = 1 - fg_a
        fg_a_expanded = np.full_like(origin, np.array(color).astype(np.float64))
        fg_c = fg_a_expanded / UINT8_MAX
        bg_c = origin.astype(np.float64) / UINT8_MAX

        fg = np.stack([fg_a, fg_a, fg_a], axis=-1) * fg_c
        bg = np.stack([bg_a, bg_a, bg_a], axis=-1) * bg_c
        # if np.any(fg != 0):
        #     print("foreground")
        return (fg + bg) * UINT8_MAX


    def get_map(self) -> NDArray[np.uint8]:
        """
        Get a color matrix representing the layout of the world
        Including water (blue), land (brown), trees (green), towns (gray) and ice (white)
        :return: 4D np.uint8 matrix representing RGBA values
        """
        water_pos = self.water.get_water_position()
        h, w = water_pos.shape
        output = np.zeros((h, w, 3), dtype=np.uint8)
        output[water_pos] = LIGHT_BLUE
        output[~water_pos] = BROWN
        output[self.forest.mat] = GREEN
        output[self.industry.mat] = RED
        if self.show_pollution_toggle:
            output = self.overlay_color(output, self.pollution.mat, PURPLE)
        return output

    def sea_level_history(self):
        return self.sea_level[:self.sea_level_ptr]

    def pollution_history(self):
        return self.pollution_tracker[:self.pollution_tracker_ptr]