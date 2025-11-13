from EcoWorldComponents import *

# constants
INITIAL_HISTORY_SIZE = 100

class EcoWorld:
    """Aggregation of all EcoWorldComponent classes into a single iterable class"""
    def __init__(self, rnd_gen:np.random.Generator, size:int = DEFAULT_SIZE):
        # Initialize components
        self.surface = Surface(rnd_gen, size)
        self.water = Water(self.surface)
        # Data trackers
        self.sea_level = np.empty(INITIAL_HISTORY_SIZE, dtype=np.int8)

    @staticmethod
    def double_array_size(a:np.typing.NDArray) -> None:
        tmp = a
        a = np.empty(a.shape[0] * 2)
        a[:tmp.shape[0]] = tmp

    def __iter__(self):
        return self

    def __next__(self):
        # advance components by one step
        next(self.water)
        return self