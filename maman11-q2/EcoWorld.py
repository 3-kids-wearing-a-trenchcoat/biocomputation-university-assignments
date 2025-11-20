from concurrent.futures import ThreadPoolExecutor

import numpy as np

from EcoWorldComponents import *
import atexit

# constants
INITIAL_HISTORY_SIZE = 100
# colors
LIGHT_BLUE = (0, 150, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
BROWN = (101, 67, 33)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
WHITE = (255, 255, 255)
OVERLAY_COLOR_ALPHA = 0.5

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
        self.temperature = Temperature(self.water, self.wind, self.pollution, self.forest)
        self.ice = Ice(self.water)
        self.ice.temperature = self.temperature # add explicitly as this couldn't be in Ice's init
        self.water.ice = self.ice
        # Data trackers
        self.tracker_ptr = 0
        # average sea level
        self.sea_level = np.zeros(INITIAL_HISTORY_SIZE, dtype=np.float16)
        # total pollution
        self.pollution_tracker = np.zeros(INITIAL_HISTORY_SIZE, dtype=np.uint32)
        # average temperature
        self.average_temperature = np.zeros(INITIAL_HISTORY_SIZE, dtype=np.float16)
        # min/max

        # initialize toggles
        self.show_surface = True
        self.show_pollution_toggle = True
        self.show_temperature_toggle = False
        self.show_clouds_toggle = False
        # initialize parallelism
        self.executor = ThreadPoolExecutor(max_workers=8)
        atexit.register(self.executor.shutdown, wait=True)

    @staticmethod
    def double_array_size(a: NDArray) -> np.typing.NDArray:
        output = np.empty(a.shape[0] * 2, dtype=a.dtype)
        output[:a.shape[0]] = a
        return output

    def update_trackers(self) -> None:
        # sea level
        self.sea_level[self.tracker_ptr] = self.water.average_sea_level()
        # pollution
        self.pollution_tracker[self.tracker_ptr] = self.pollution.get_total_pollution()
        # temperature
        self.average_temperature[self.tracker_ptr] = self.temperature.get_average()
        # update tracker ptr and resize if needed
        self.tracker_ptr += 1
        if self.tracker_ptr == self.sea_level.shape[0]:
            self.sea_level = self.double_array_size(self.sea_level)
            self.pollution_tracker = self.double_array_size(self.pollution_tracker)
            self.average_temperature = self.double_array_size(self.average_temperature)

    def update_component_pointers(self):
        self.wind.update_components(self.water)
        self.forest.update_components(self.water)
        self.industry.update_components(self.water)
        self.pollution.update_components(self.industry, self.forest, self.wind)
        self.temperature.update_components(self.water, self.wind, self.pollution, self.forest)

    def __iter__(self):
        return self

    def __next__(self):
        # advance components by one step
        comp_names = ["water", "wind", "forest", "industry", "pollution", "temperature"]
        futures = {name: self.executor.submit(next, getattr(self, name))
                   for name in comp_names}
        for name, fut in futures.items():
            setattr(self, name, fut.result())

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
                      max_point=UINT8_MAX, proportional:bool=True) -> NDArray:
        if not proportional:
            return np.where(mask,
                            ((1 - OVERLAY_COLOR_ALPHA)*origin) + OVERLAY_COLOR_ALPHA * np.full_like(origin, color),
                            origin)
        fg_a = mask.astype(np.float64) / max_point
        bg_a = 1 - fg_a
        fg_a_expanded = np.full_like(origin, np.array(color).astype(np.float64))
        fg_c = fg_a_expanded / max_point
        bg_c = origin.astype(np.float64) / max_point

        fg = np.stack([fg_a, fg_a, fg_a], axis=-1) * fg_c
        bg = np.stack([bg_a, bg_a, bg_a], axis=-1) * bg_c
        return ((fg + bg) * max_point).astype(np.uint8)


    def get_map(self) -> NDArray[np.uint8]:
        """
        Get a color matrix representing the layout of the world
        Including water (blue), land (brown), trees (green), towns (gray) and ice (white)
        :return: 4D np.uint8 matrix representing RGBA values
        """

        h, w = self.surface.mat.shape
        if self.show_surface:
            water_pos = self.water.get_water_position()
            output = np.zeros((h, w, 3), dtype=np.uint8)
            output[water_pos] = LIGHT_BLUE
            output[~water_pos] = BROWN
            output[self.ice.mat > 0] = WHITE
            if not self.show_temperature_toggle:
                # blending the colors with green and red makes the temperature read difficult
                output[self.forest.mat] = GREEN
                output[self.industry.mat] = RED
        else:
            output = np.full((h, w, 3), BLACK, dtype=np.uint8)
        if self.show_temperature_toggle:
            above_zero = np.where(self.temperature.temp > 0, self.temperature.temp, 0)
            below_zero = np.where(self.temperature.temp < 0, -self.temperature.temp, 0)
            max_temp = INT8_MAX // 4
            output = self.overlay_color(output, above_zero, RED, max_temp)
            output = self.overlay_color(output, below_zero, BLUE, max_temp)
        if self.show_pollution_toggle:
            output = self.overlay_color(output, self.pollution.mat, PURPLE)
        return output

    # value histories (for plotting)
    def sea_level_history(self):
        return self.sea_level[:self.tracker_ptr]

    def pollution_history(self):
        return self.pollution_tracker[:self.tracker_ptr]

    def temperature_history(self):
        return self.average_temperature[:self.tracker_ptr]

    # toggle functions tied to GUI checkboxes and buttons
    def pollution_toggle(self, checked: bool):
        self.show_pollution_toggle = checked

    def temperature_toggle(self, checked: bool):
        self.show_temperature_toggle = checked

    def clouds_toggle(self, checked: bool):
        self.show_clouds_toggle = checked

    def surface_toggle(self, checked: bool):
        self.show_surface = checked