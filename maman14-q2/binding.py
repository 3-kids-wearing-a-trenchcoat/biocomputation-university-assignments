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
# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Iterable, cast
import strand
# from external_njit_functions import choose_binding_by_strength
from LazyLock import LazyLock
from functools import partial
import random

# constants
# MIN_OVERLAP:int = 3
SEED_LEN = 6
THREADS = 8     # maximum number of threads running concurrently when manually parallelizing work
ANNEALING_FAILURE_PROB = 1e-5   # probability that two strands wouldn't anneal despite being neighbors
rng = np.random.default_rng()
FLOAT_DTYPE = np.float32

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
        # TODO: If I were to change strength from fraction of total nucleotides that are bound, I need to change it here too
        seq_a, seq_b = strand.get_seq(id_a), strand.get_seq(id_b)
        bind_seq_a, bind_seq_b = seq_a[start_a: start_a + length], seq_b[start_b: start_b + length]
        comp = (bind_seq_a ^ bind_seq_b).count()
        strength = comp / (len(seq_a) + len(seq_b))

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
    # Normalize strength to be used as probability
    probs = np.asarray([entry[3] for entry in candidates], dtype=FLOAT_DTYPE)
    total = probs.sum()
    # TODO: If I wanted to apply some probability function like strength**2, this would be the place
    probs = probs / total
    chosen = int(rng.choice(len(candidates), p=probs))
    return candidates[chosen]


# def _validate_bind(strand_locks: LazyLock, id_a:int, start_a:int, id_b:int, start_b:int, length:int) -> bool:
#     if id_a < id_b:
#         first_lock, second_lock = strand_locks[id_a], strand_locks[id_b]
#     else:
#         first_lock, second_lock = strand_locks[id_b], strand_locks[id_a]
#     first_lock.acquire()
#     second_lock.acquire()
#     conflict = (not is_bound_at(id_a, start_a, length)) and (not is_bound_at(id_b, start_b, length))
#     second_lock.release()
#     first_lock.release()
#     return conflict

def _validate_bind(id_a:int, start_a:int, id_b:int, start_b:int, length:int) -> bool:
    # This function only reads, and modifications only happen after all _validate_bind are done, so no locks needed here
    # check if strand A or strand B are already bound at their intended binding span
    if is_bound_at(id_a, start_a, length) or is_bound_at(id_b, start_b, length):
        return False
    # check if A and B are already bound to one another at any point
    if ((_A_id == id_a & _B_id == id_b) | (_A_id == id_b & _B_id == id_a)).any():
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
        # with ThreadPoolExecutor(max_workers=THREADS) as ex:
        with ProcessPoolExecutor(max_workers=THREADS) as ex:
            binds: List[Tuple[int, int, int, float]] = list(ex.map(_choose_binding, candidates, choices))
        # For each pair and their chosen binding, check if the binding area in either strand is not already bound
        # If it is, that binding is discarded
        strand_locks = LazyLock()                                   # per-strand locks
        # f = partial(_validate_bind, strand_locks=strand_locks)      # map strand_locks to _validate_bind
        start_a, start_b, bind_length, strength = zip(*binds)
        # with ThreadPoolExecutor(max_workers=THREADS) as ex:
        with ProcessPoolExecutor(max_workers=THREADS) as ex:
            bind_is_valid =list(ex.map(_validate_bind, candidates, start_a, choices, start_b, bind_length))
        # add all found bindings that are possible
        [add_bind(candidates[i], start_a[i], choices[i], start_b[i], bind_length[i], strength[i])
         for i in range(len(start_a)) if bind_is_valid[i]] # TODO: BAD AND SLOW

@njit
def get_bound_strands(host_id:int, sort_by_start:bool = True) -> Tuple[NDArray[np.uint32], NDArray[np.uint8],
                                                                                                NDArray[np.uint8]]:
    """
    Get all strands bound to the input strand
    :param host_id: id of the host strand ("host" being the strand all the output strands are bound to)
    :param sort_by_start: Whether to sort the output arrays by start position
    :return: Tuple of 3 numpy arrays, they are (in order)
             1. array of IDs of strands bound to the host (np.uint32)
             2. array of binding start position relative to host (np.uint8)
             3. array of binding lengths (np.uint8)
    """
    # TODO: It may be smart to instead have some kind of hash map instead of recreate this at every annealing
    host_id_in_A = (_A_id == host_id)
    host_id_in_B = (_B_id == host_id)

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

@njit
def _keep_only_unique(pairs: NDArray, hosts: NDArray|None = None) -> NDArray|Tuple[NDArray, NDArray]:
    """
    Given a numpy array of shape (N,2) (effectively an array of tuples of size 2), remove from the array
    all tuples which contain a value that appeared in a previous tuple and return it.
    """
    if pairs.shape[1] != 2:
        raise ValueError("`pairs` must be an (N,2) shaped NDArray")
    N = pairs.shape[0]      # number of pairs
    flat = pairs.ravel()    # flatten pairs
    # unique_vals is a list of the unique values found
    # first_flat_idx matches - for every cell i in flat - the index in unique_vals containing its value
    unique_vals, first_flat_idx = np.unique(flat, return_index=True)
    first_row_of_unique = first_flat_idx // 2   # row of first appearance

    # map each flat value to index in unique_vals
    idx_into_unique = np.searchsorted(unique_vals, flat)
    # get for each flat position the row of first instance
    first_row_flat = first_row_of_unique[idx_into_unique]   # length 2*N
    # reshape back to (N,2) and require both elements' first_row == current_row
    first_row_pairs = first_row_flat.reshape(N,2)
    row_indices = np.arange(N)[:,None]
    keep_mask = np.all(first_row_pairs == row_indices, axis=1)  # shape (N,)

    if hosts is None:
        return pairs[keep_mask]
    return pairs[keep_mask], hosts[keep_mask]

def _merge_strands(id_a:int, id_b:int, host_id:int) -> None:
    """
    Unite strand A and strand B by appending strand B to strand A.
    Will update the strand data as well as the binding data.
    :param id_a: id of strand A
    :param id_b: id of strand B, which will be appended to the end of strand A
    :param host_id: id of host strand. The host strand is the strand to which A and B are connected in such a way
                    that they can be merged.
    """
    # TODO: If this were part of a parallelized process, this should use locks of some kind.
    new_id = strand.merge_strands(id_a, id_b)       # create merged strand and delete the two input strands
    # create masks for which rows reference id_a / id_b on each side (excluding inactive binds)
    a_in_A = (_A_id == id_a) & _active
    a_in_B = (_B_id == id_a) & _active
    b_in_A = (_A_id == id_b) & _active
    b_in_B = (_B_id == id_b) & _active
    # find and exclude host bind from the above (this will be explicitly handled now)
    host_bind_mask = ((a_in_A | b_in_A) & (_B_id == host_id)) | ((a_in_B | b_in_B) & (_A_id == host_id))
    assert np.count_nonzero(host_bind_mask) == 2    # sanity check, should be host bound to a and host bound to b, no more
    binds_with_host = np.nonzero(host_bind_mask)[0]    # get indexes of host bind
    a_in_A[binds_with_host], a_in_B[binds_with_host], b_in_A[binds_with_host], b_in_B[binds_with_host] = False
    _active[binds_with_host[0]], _active[binds_with_host[1]] = 0, 0 # delete binds featuring host

    

# TODO: Bulk annealing (stochastic)
def bulk_anneal() -> None:
    """Anneal all neighboring strands that are bound to the same third strand with a small probability of failure."""
    # get ids, start position (relative to host) and bind lengths each strand bound to host
    bound_ids, bind_start = np.array([], dtype=np.uint32), np.array([], dtype=np.uint8)
    length, hosts = np.array([], dtype=np.uint8), np.array([], dtype=np.uint32)
    for host_id in strand.get_living_ids():
        # bound_ids, bind_start, length = get_bound_strands(host_id)
        func_output = get_bound_strands(host_id)
        if not func_output[0]:  # if host has no bindings
            continue            # skip
        bound_ids = np.append(bound_ids, func_output[0])
        bind_start = np.append(bind_start, func_output[1])
        length = np.append(length, func_output[2])
        hosts = np.append(hosts, np.full(bound_ids.size, host_id))
    # Candidates are those whose end position is immediately before the another's start position
    candidate_mask = (bind_start[:-1] + length[:-1]) == bind_start[1:]
    candidates = np.column_stack((bound_ids[:-1][candidate_mask], bound_ids[1:][candidate_mask]))
    # Filter the candidates list so that no value in any pair appears in an earlier pair.
    # This is to avoid having to concatenate anneals. that is, new strands produced in this run of bulk_anneal
    # will not merge with any other strand during this run.
    candidates, host = _keep_only_unique(candidates, hosts)
    # TODO: the following section could probably be parallelized
    [_merge_strands(pair[0], pair[1], host) for pair, host in zip(candidates, host)]


# TODO: bulk unravel binds at random, as a function of global temperature and per-bind strength
# TODO: Implement restriction enzymes
# TODO: Magnetic separation

# TODO: Multithreading takes more work than I expected, this is a LOWER PRIORITY to the rest of the project
# TODO: added @njit wherever possible (the functions here are mostly numpy stuff, this could work)