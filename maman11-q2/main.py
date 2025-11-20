import gui
import EcoWorld as ew
import numpy.random

# constants
SEED = 1234
RNG = numpy.random.default_rng(SEED)

if __name__ == "__main__":
    world = ew.EcoWorld(RNG)
    gui.run(world)