from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from threading import RLock
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Iterable
import strand
from LazyLock import LazyLock
from binding import (_A_id, _B_id, _A_start, _B_start, _length, _active, _strength)

# constants
# temperature
UNRAVEL_TEMP = 95       # temperature at which we set the experiment when we want strands to unravel (max temperature)
BIND_TEMP = 55          # temperature at which we set the experiment when we want strands to bind (min temperature)
MID_POINT = (UNRAVEL_TEMP + BIND_TEMP) / 2
# probability function
PROB_FLOOR = 1e-4       # probability of strongest possible bind to remain bound under UNRAVEL_TEMP (min prob of unraveling)
STEEPNESS = 4.0         # steepness of change between temperatures, smaller values = sharper transitions

# functions
def get_unravel_probability(temperature:float, strengths:NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Apply a sigmoidal function on strengths whose output is the probability of a bind of this strength ot unravel.
    Output is in the range [PROB_FLOOR, 1], the stronger the bind, the lower its probability to unravel will be.
    The higher the temperature, the more likely a strand is to unravel.
    """
    melt = 1 / (1 + np.exp(- ((temperature - MID_POINT) / STEEPNESS)))
    remain_prob = strengths * (1 - (1 - STEEPNESS) * melt)
    return 1 - remain_prob

def select_binds_to_unravel(temperature:float) -> NDArray[np.uint32]:
    """
    Select which binds to unravel as a function of temperature and strength
    :param temperature: temperature the sample is set to
    :return: ordered indexes of binds to unravel
    """
    active_strengths = _strength[_active]                                         # get strengths of active binds
    active_unravel_probs = get_unravel_probability(temperature, active_strengths) # calc prob to unravel for each strand
    # According to the probabilities, choose which strands to unravel
    rand = np.random.rand(len(active_strengths))
    unravel_mask = rand <= active_unravel_probs
    # according to the unraveling mask, return indexes of binds to unravel
    active_idx = np.nonzero(_active)
    return active_idx[unravel_mask]




