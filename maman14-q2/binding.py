"""
This module represents the bindings of strands into a double-helix structure.
The data is stored as a group of numpy arrays which represent the fields of a table, with one entry for each binding.
**id** -- not an explicit field, represented by the entry index
**A_id** -- id (according to strand.py) of one member of a bind
**A_start** -- The start position of the bind relative to A
**B_id** -- id (according to strand.py) of the second member of a bind
**B_start** -- The start position of the bind relative to B
**length** -- The length of the bind, so B covers the **length** nucleotides of A starting from **A_start** and
              A covers the **length** nucleotides of B starting from **B_start**
**strength** -- Strength of bind. Determined by the number of complementary nucleotides actually forming the bond.
                Binds un-bind spontaneously depending on temperature and **strength**, the stronger the bind is,
                the less likely it is to actually un-bind
**active** -- Whether the bind actually exists, used as a soft-delete.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from threading import RLock
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Iterable, cast
import strand
from external_njit_functions import choose_binding_by_strength
from LazyLock import LazyLock
from functools import partial
import random

# constants
# MIN_OVERLAP:int = 3
SEED_LEN = 6
THREADS = 8     # maximum number of threads running concurrently when manually parallelizing work

# binding data
_A_id = np.empty(0, dtype=np.uint32)
_A_start = np.empty(0, dtype=np.uint8)
_B_id = np.empty(0, dtype=np.uint32)
_B_start = np.empty(0, dtype=np.uint8)
_length = np.empty(0, dtype=np.uint8)
_strength = np.empty(0, dtype=np.float16)
_active:NDArray[np.bool] = np.array(0, dtype=np.bool)   # active binds
_lock = RLock() # avoid race conditions as a result of parallel writes

# functions

def is_bound_at(strand_id:int, bind_start:int, bind_length:int) -> bool:
    """
    Check if the given strand is bound within the given range
    :param strand_id: id of the input strand to check
    :param bind_start: Start checking for binding from this index (inclusive)
    :param bind_length: Length of sequence to check starting from bind_start
    :return: 'True' if this strand is bound within the given range, otherwise 'False'
    """
    bind_end = bind_start + bind_length - 1 # end index (inclusive) of input bind
    # Select entries that are active and whose id (in field A or B) match strand_ID
    select_in_A, select_in_B = _active & _A_id == strand_id, _active & _B_id == strand_id
    # Select start values of relevant entries
    start_in_A, start_in_B = _A_start[select_in_A], _B_start[select_in_B]
    # get end values (inclusive) of relevant entries
    end_in_A, end_in_B = start_in_A + _length[select_in_A] - 1, start_in_B + _length[select_in_B] - 1
    # check if start <= bind_start <= end or start <= bind_end <= end in any index of A or B
    return  ((start_in_A <= bind_start & bind_start <= end_in_A).any() or
             (start_in_A <= bind_end   & bind_end <= end_in_A).any()   or
             (start_in_B <= bind_start & bind_start <= end_in_B).any() or
             (start_in_B <= bind_end   & bind_end <= end_in_B).any())

def add_bind(id_a:int, start_a:int, id_b:int, start_b:int, length:int, strength:float) -> int:
    """
    Create a new duplex binding.
    THIS FUNCTION DOES NOT CHECK FOR OVERLAP WITH ANY EXISTING BINDING
    :param id_a: id of strand A
    :param start_a: starting bind index at strand A
    :param id_b: id of strand B
    :param start_b: starting bind index at strand B
    :param length: binding length
    :param strength: binding strength
    :return: id of new binding
    """
    global _A_id, _B_id, _active
    _lock.acquire()
    new_id = len(_A_id)
    _A_id, _B_id = np.append(_A_id, id_a), np.append(_B_id, id_b)
    _active = np.append(_active, True)
    _lock.release()
    return new_id

def delete_bind(bind_id:int) -> None:
    _lock.acquire()
    _active[bind_id] = False
    _lock.release()

def reindex() -> None:
    # TODO: I THINK there's no need to inform anything of the changes in ID, if I'm wrong I need to return a conversion array a la strand's reindex
    """
    Compactify data by removing all inactive strands
    :return: None
    """
    global _A_id, _B_id, _A_start, _B_start, _length, _active, _strength
    _lock.acquire()
    _A_id, _B_id, _A_start, _B_start, _length, _strength = (_A_id[_active], _B_id[_active], _A_start[_active],
                                                            _B_start[_active], _length[_active], _strength[_active])
    _active = np.ones(len(_A_id), dtype=np.bool)
    _lock.release()


# TODO: get bind (should think about what it actually returns)
# TODO: added @njit wherever possible (the functions here are mostly numpy stuff, this could work)

def _choose_binding(id_a:int, id_b:int) -> Tuple[int, int, int, float]|None:
    """
    Get a possible binding for strands A and B.
    This binding is chosen randomly, with probabilities weighted by binding strength.
    :param id_a: id of strand A
    :param id_b: is of strand B
    :return: If no possible binding exists, returns None.
             Otherwise, return a tuple made up of the following values in order:
             1. Bind start position for strand A (int)
             2. Bind start position for strand B (int)
             3. Bind length (int)
             4. Bind strength - fraction of total nucleotides in both strands bound to a complementary nucleotide (float)
    """
    # generate candidate bindings
    candidates:List[Tuple[int, int, int, float]] = strand.get_possible_binds(id_a, id_b, SEED_LEN)
    if not candidates:
        return None         # if candidates is empty, no bindings are possible, stop here and return 'None'
    return choose_binding_by_strength(candidates)

def _validate_bind(strand_locks: LazyLock, id_a:int, start_a:int, id_b:int, start_b:int, length:int) -> bool:
    # TODO: implement

def bulk_bind(repetitions:int=10) -> None:
    """
    Randomly bind strands to one another.
    This function mimics (or should, at least) the natural binding behavior of DNA strands.
    At each iteration, only some of the strands will form new bonds
    :param repetitions: number of times to repeat this process
    """
    for rep in range(repetitions):  # repeat the bulk bind process `repetitions` times
        # list of strands that will choose a partner (initialized to active ones)
        candidates = np.nonzero(_active)[0].astype(np.uint32)
        n = candidates.size
        if n < 2:   # if n == 0 or n == 1, no further bindings are possible
            return
        # each candidate chooses a *different* candidate at random, using the vectorized shift trick
        r = np.random.randint(0, n - 1, size=n) # generate random integers from range [0, n-2]
        # convert each r into an index by skipping one spot when r >= i
        i = np.arange(n)
        idx = r + (r >= i)
        # convert random idx choices into a random value from candidates, importantly, one that isn't the same strand.
        choices = candidates[idx].astype(candidates.dtype)

        # For each candidate-choice pair, pick a binding at random weighted by strength
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            binds: List[Tuple[int, int, int, float]] = list(ex.map(_choose_binding, candidates, choices))
        # For each pair and their chosen binding, check if the binding area in either strand is not already bound
        # If it is, that binding is discarded
        strand_locks = LazyLock()                                   # per-strand locks
        f = partial(_validate_bind, strand_locks=strand_locks)      # map strand_locks to _validate_bind
        start_a, start_b, bind_length, strength = zip(*binds)
        with ThreadPoolExecutor(max_workers=THREADS) as ex:
            bind_is_valid =list(ex.map(_validate_bind, candidates, start_a, choices, start_b, bind_length))
        # add all found bindings that are possible
        # TODO: THIS IMPLEMENTATION IS ABSOLUTELY DISGUSTING, parallelize it. (probably by using LazyLock in all functions somehow)
        [add_bind(candidates[i], start_a[i], choices[i], start_b[i], bind_length[i], strength[i])
         for i in range(len(start_a)) if bind_is_valid[i]]

# TODO: bulk unravel binds at random, as a function of global temperature and per-bind strength
# TODO: Bulk annealing (stochastic)
# TODO: Implement restriction enzymes
# TODO: Magnetic separation