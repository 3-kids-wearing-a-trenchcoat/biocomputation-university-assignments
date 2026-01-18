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
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Iterable
import strand
from LazyLock import LazyLock

# constants
# MIN_OVERLAP:int = 3
SEED_LEN = 6
THREADS = 8     # maximum number of threads running concurrently when manually parallelizing work
rng = np.random.default_rng()
FLOAT_DTYPE = np.float32

# binding data
_A_id = np.array([], dtype=np.uint32)
_A_start = np.array([], dtype=np.uint16)
_B_id = np.array([], dtype=np.uint32)
_B_start = np.array([], dtype=np.uint16)
_length = np.array([], dtype=np.uint16)
_strength = np.array([], dtype=np.float16)
_active:NDArray[np.bool] = np.array([], dtype=np.bool)   # active binds
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
    # TODO: could probably be vectorized
    bind_end = bind_start + bind_length - 1 # end index (inclusive) of input bind
    # Select entries that are active and whose id (in field A or B) match strand_ID
    select_in_A, select_in_B = _active & _A_id == strand_id, _active & _B_id == strand_id
    # Select start values of relevant entries
    start_in_A, start_in_B = _A_start[select_in_A], _B_start[select_in_B]
    # get end values (inclusive) of relevant entries
    end_in_A, end_in_B = start_in_A + _length[select_in_A] - 1, start_in_B + _length[select_in_B] - 1
    # check if start <= bind_start <= end or start <= bind_end <= end in any index of A or B
    return  (((start_in_A <= bind_start) & (bind_start <= end_in_A)).any() or
             ((start_in_A <= bind_end)   & (bind_end <= end_in_A)).any()   or
             ((start_in_B <= bind_start) & (bind_start <= end_in_B)).any() or
             ((start_in_B <= bind_end)   & (bind_end <= end_in_B)).any())

def calc_strength(id_a:int|NDArray, start_a:int|NDArray,
                  id_b:int|NDArray, start_b:int|NDArray, length:int|NDArray) -> float|NDArray[np.float32]:
    if type(id_a) == int:   # if input is for a single bind
        seq_a, seq_b = strand.get_seq(id_a), strand.get_seq(id_b)
        bind_seq_a, bind_seq_b = seq_a[start_a: start_a + length], seq_b[start_b: start_b + length]
        comp = (bind_seq_a ^ bind_seq_b).count()
        # return (comp * 2) / (len(seq_a) + len(seq_b))
        return comp / length

    # if array of binds (assumed to be of the same length and aligned by id)
    length_arr = strand.get_length()
    len_a, len_b = length_arr[id_a], length_arr[id_b]   # length of strand A and strand B respectively
    # TODO: disgustingly inefficient, reconsider
    comp = np.array([(strand.get_seq(a)[start_a[a]: start_a[a] + len_a[a]] ^
                      strand.get_seq(b)[start_b[b]: start_b[b] + len_b[b]]).count()
                      for a, b in zip(id_a, id_b)], dtype=np.uint16)
    return ((comp * 2) / (len_a + len_b)).astype(np.float32)


def add_bind(id_a:int, start_a:int, id_b:int, start_b:int, length:int, strength:float|None=None) -> int:
    """
    Create a new duplex binding.
    THIS FUNCTION DOES NOT CHECK FOR OVERLAP WITH ANY EXISTING BINDING
    :param id_a: id of strand A
    :param start_a: starting bind index at strand A
    :param id_b: id of strand B
    :param start_b: starting bind index at strand B
    :param length: binding length
    :param strength: optional - binding strength. If not given, will be calculated.
    :return: id of new binding
    """
    global _A_id, _B_id, _A_start, _B_start, _length, _active, _strength
    # If strength isn't given, calculate it
    if strength is None:
        strength = calc_strength(id_a, start_a, id_b, start_b, length)
    _lock.acquire()
    new_id = len(_A_id)
    _A_id, _B_id = np.append(_A_id, id_a), np.append(_B_id, id_b)
    _A_start, _B_start = np.append(_A_start, start_a), np.append(_B_start, start_b)
    _length = np.append(_length, length)
    _active = np.append(_active, True)
    _strength = np.append(_strength, strength)
    _lock.release()
    return new_id

def delete_bind(bind_id:int) -> None:
    _lock.acquire()
    _active[bind_id] = False
    _lock.release()

def get_dead_fraction() -> float:
    """Get fraction of strand entries which are dead"""
    return np.count_nonzero(_active) / len(_active)

def reindex(new_strand_ids:NDArray[np.int64]|None = None) -> None:
    """
    Compactify data by removing all inactive strands
    :param new_strand_ids: Numpy array produced by strand's reindex. If this variable is provided, strand IDs
    are updated to the new values after the binding database is compressed.
    Otherwise, the binding database is simply compressed by deleting all inactive entries.
    """
    global _A_id, _B_id, _A_start, _B_start, _length, _active, _strength
    _lock.acquire()
    # TODO: haven't decided if I need to lock strand's lock here, or if locks are even necessary at all in retrospect
    _A_id, _B_id, _A_start, _B_start, _length, _strength = (_A_id[_active], _B_id[_active], _A_start[_active],
                                                            _B_start[_active], _length[_active], _strength[_active])
    _active = np.ones(len(_A_id), dtype=np.bool)

    if new_strand_ids is not None:                  # update strand IDs
        for i in range(len(_A_id)):                 # for every entry
            _A_id[i] = new_strand_ids[_A_id[i]]     # change _A_id values according to new_strand_ids
            _B_id[i] = new_strand_ids[_B_id[i]]     # change _B_id values according to new strand_ids

    _lock.release()

def _bulk_add(id_a: NDArray, start_a: NDArray, id_b: NDArray, start_b: NDArray,
              length: NDArray, strength:NDArray|None = None) -> None:
    global _A_id, _A_start, _B_id, _B_start, _length, _strength, _active
    if strength is None:
        strength = calc_strength(id_a, start_a, id_b, start_b, length)
    _lock.acquire()
    _A_id = np.concatenate((_A_id, id_a))
    _A_start = np.concatenate((_A_start, start_a))
    _B_id = np.concatenate((_B_id, id_b))
    _B_start = np.concatenate((_B_start, start_b))
    _length = np.concatenate((_length, length))
    _strength = np.concatenate((_strength, strength))
    _active = np.concatenate((_active, np.ones(len(id_a), dtype=np.bool)))
    _lock.release()

# ==========BINDING FUNCTIONS==========

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
             4. Bind strength - fraction of nucleotides in bound span that are complementary
    """
    # generate candidate bindings
    candidates:List[Tuple[int, int, int, float]] = strand.get_possible_binds(id_a, id_b, SEED_LEN)
    if not candidates:
        return None         # if candidates is empty, no bindings are possible, stop here and return 'None'
    # Normalize strength to be used as probability
    probs = np.asarray([entry[3] for entry in candidates], dtype=FLOAT_DTYPE)
    total = probs.sum()
    # TODO: If I wanted to apply some probability function like strength**2, this would be the place
    probs = probs / total
    chosen = int(rng.choice(len(candidates), p=probs))
    return candidates[chosen]

def _validate_bind(id_a:int, start_a:int, id_b:int, start_b:int, length:int) -> bool:
    # This function only reads, and modifications only happen after all _validate_bind are done, so no locks needed here
    # check if strand A or strand B are already bound at their intended binding span
    if is_bound_at(id_a, start_a, length) or is_bound_at(id_b, start_b, length):
        return False
    # check if A and B are already bound to one another at any point
    if (((_A_id == id_a) & (_B_id == id_b)) | ((_A_id == id_b) & (_B_id == id_a))).any():
        return False
    return True

def bulk_bind(repetitions:int=10) -> None:
    """
    Randomly bind strands to one another.
    This function mimics (or should, at least) the natural binding behavior of DNA strands.
    At each iteration, only some of the strands will form new bonds
    :param repetitions: number of times to repeat this process
    """
    for rep in range(repetitions):  # repeat the bulk bind process `repetitions` times
        # list of strands that will choose a partner (initialized to active ones)
        # candidates = np.nonzero(_active)[0].astype(np.uint32)
        candidates = np.nonzero(strand.get_active_mask())[0].astype(np.uint32)
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
        # with ProcessPoolExecutor(max_workers=THREADS) as ex:
        #     binds: List[Tuple[int, int, int, float]] = list(ex.map(_choose_binding, candidates, choices))
        binds: List[Tuple[int, int, int, float]] = list(map(_choose_binding, candidates, choices))
        # For each pair and their chosen binding, check if the binding area in either strand is not already bound
        # If it is, that binding is discarded
        # strand_locks = LazyLock()                                   # per-strand locks
        # f = partial(_validate_bind, strand_locks=strand_locks)      # map strand_locks to _validate_bind
        start_a, start_b, bind_length, strength = binds[0], binds[1], binds[2], binds[3]
        # with ProcessPoolExecutor(max_workers=THREADS) as ex:
        #     bind_is_valid =list(ex.map(_validate_bind, candidates, start_a, choices, start_b, bind_length))
        if _active.length > 0:
            bind_is_valid = list(map(_validate_bind, candidates, start_a, choices, start_b, bind_length))
        # add all found bindings that are possible
        [add_bind(candidates[i], start_a[i], choices[i], start_b[i], bind_length[i], strength[i])
         for i in range(len(start_a)) if bind_is_valid[i]] # TODO: BAD AND SLOW

# @njit
def get_bound_strands(host_id:int, sort_by_start:bool = True) -> Tuple[NDArray[np.uint32], NDArray[np.uint16],
                                                                                                NDArray[np.uint16]]:
    """
    Get all strands bound to the input strand
    :param host_id: id of the host strand ("host" being the strand all the output strands are bound to)
    :param sort_by_start: Whether to sort the output arrays by start position
    :return: Tuple of 3 numpy arrays, they are (in order)
             1. array of IDs of strands bound to the host (np.uint32)
             2. array of binding start position relative to host (np.uint16)
             3. array of binding lengths (np.uint16)
    """
    # TODO: It may be smart to instead have some kind of hash map instead of recreate this at every annealing
    host_id_in_A = (_A_id == host_id) & _active
    host_id_in_B = (_B_id == host_id) & _active

    bound_ids1, bound_ids2 = _B_id[host_id_in_A], _A_id[host_id_in_B]
    bound_ids = np.concatenate((bound_ids1, bound_ids2))
    bind_start1, bind_start2 = _A_start[host_id_in_A], _B_start[host_id_in_B]
    bind_start = np.concatenate((bind_start1, bind_start2))
    length = _length[host_id_in_A | host_id_in_B]

    if sort_by_start:
        start_sorted_indexes = np.argsort(bind_start)
        bound_ids, bind_start, length = (bound_ids[start_sorted_indexes], bind_start[start_sorted_indexes],
                                         length[start_sorted_indexes])

    return bound_ids, bind_start, length

def get_bound_ids(host_id:int|NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Get just the IDs of strands bound to host_id"""
    host_id_in_A_mask = np.isin(_A_id, host_id) & _active
    host_id_in_B_mask = np.isin(_B_id, host_id) & _active
    bound_ids_a = _B_id[host_id_in_A_mask]
    bound_ids_b = _A_id[host_id_in_B_mask]
    # return np.unique(np.concatenate((bound_ids_a, bound_ids_b)))
    return np.union1d(bound_ids_a, bound_ids_b)

def is_bound(strand_id: int|NDArray[np.uint32]) -> bool|NDArray[np.bool]:
    """
    Check if strands are bound to any other strand. (inactive binds excluded)
    :param strand_id: single ID or NDArray of IDs for strands to check
    :return: if strand_id is a single int, return 'True' if it's bound, otherwise 'False.
             if strand_id is an NDArray, return a boolean NDArray where index `i` equals
             'True' iff `strand_id[i]` is bound to some other strand.
    """
    vals = np.union1d(_A_id[_active], _B_id[_active])   # sorted list of all ids that are bound
    return np.isin(strand_id, vals) # for every strand_id, return if it's in vals

def get_binds_that_contain(strand_id: NDArray[np.uint32]) -> NDArray[np.uint32]:
    """Get IDs of all binds that contain one of the input strand IDs"""
    mask_a, mask_b = np.isin(_A_id, strand_id), np.isin(_B_id, strand_id)
    idx_a, idx_b = np.nonzero(mask_a & _active), np.nonzero(mask_b & _active)
    return np.union1d(idx_a, idx_b).astype(np.uint32)

def delete_all_with_strand_id(strand_id: int|NDArray[np.uint32]) -> None:
    """
    Delete all binds that contain the input strand(s) or do NOT contain it/them
    :param strand_id: int representing a single strand ID or an NDArray of strand IDs
    """
    _lock.acquire()
    # _active[strand_id] = False
    to_delete = get_binds_that_contain(strand_id)
    _active[to_delete] = False
    _lock.release()

def get_entry_num() -> int:
    return len(_active)

def get_active_num() -> int:
    return np.count_nonzero(_active).astype(int)

def get_all_bound_strands() -> NDArray[np.uint32]:
    """Get IDs of all strands that are bound (in order)"""
    return np.union1d(_A_id[_active], _B_id[_active])