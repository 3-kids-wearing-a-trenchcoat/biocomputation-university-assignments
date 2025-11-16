from concurrent.futures import ThreadPoolExecutor
from EcoWorldComponents import *
import atexit

# constants
INITIAL_HISTORY_SIZE = 100
# colors
LIGHT_BLUE = (0, 150, 255, 255)
BLACK = (0, 0, 0, 255)
BROWN = (101, 67, 33, 255)
GREEN = (0, 255, 0, 255)
RED = (255, 0, 0, 255)

class EcoWorld:
    """Aggregation of all EcoWorldComponent classes into a single iterable class"""
    def __init__(self, rnd_gen:np.random.Generator, size:int = DEFAULT_SIZE):
        # Initialize components
        self.surface = Surface(rnd_gen, size)
        self.water = Water(self.surface)
        self.wind = Wind(rnd_gen, self.water)
        self.forest = Forest(rnd_gen, self.surface, self.water)
        self.industry = Industry(rnd_gen, self.surface, self.water, self.forest)
        # Data trackers
        self.sea_level = np.zeros(INITIAL_HISTORY_SIZE, dtype=np.int8)
        self.sea_level_ptr = 0
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

    def update_component_pointers(self):
        self.wind.update_components(self.water)
        self.forest.update_components(self.water)
        self.industry.update_components(self.water)

    def __iter__(self):
        return self

    def __next__(self):
        comp_names = ["water", "wind", "forest", "industry"]
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

    def get_map(self) -> NDArray[np.uint8]:
        """
        Get a color matrix representing the layout of the world
        Including water (blue), land (brown), trees (green), towns (gray) and ice (white)
        :return: 4D np.uint8 matrix representing RGBA values
        """
        water_pos = self.water.get_water_position()
        h, w = water_pos.shape
        output = np.zeros((h, w, 4), dtype=np.uint8)
        output[water_pos] = LIGHT_BLUE
        output[~water_pos] = BROWN
        output[self.forest.mat] = GREEN
        output[self.industry.mat] = RED
        return output

    def sea_level_history(self):
        return self.sea_level[:self.sea_level_ptr]