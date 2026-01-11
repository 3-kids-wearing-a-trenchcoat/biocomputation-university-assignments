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
import strand
from TwoBitArray import TwoBitArray
from bitarray import bitarray

# constants
# MIN_OVERLAP:int = 3
SEED_LEN = 6

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
    _lock.acquire()
    new_id = len(_A_id)
    np.append(_A_id, id_a), np.append(_B_id, id_b)
    np.append(_active, True)
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

# TODO: