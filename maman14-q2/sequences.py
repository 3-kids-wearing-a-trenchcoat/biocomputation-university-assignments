"""
This module is an abstraction for the list of all DNA sequences
All DNA sequences are stored in a packed (sequential) manner in a TwoBitArray.
Each individual strand will be represented by the tuple (ID, offset, length, active) and will use the functions
defined here to find the range nucleotides that actually represents it.
"""

from __future__ import annotations
from TwoBitArray import TwoBitArray
from bitarray import bitarray
from typing import Iterable, Tuple, List
from threading import RLock

_nucleotide_syntax = ['T', 'G', 'C', 'A'] # 'T':=00, 'G':=01, 'C':=10, 'A':=11
_seq = TwoBitArray(_nucleotide_syntax)    # The logical array in which all DNA sequences are stored
_active = bitarray()                      # bitmap representing whether the sequence is deleted (equals 0)
_lock = RLock()                           # Prevent race-conditions as a result of parallel writing

def seq_append(input_seq: TwoBitArray|Iterable) -> Tuple[int, int]:
    """
    Store a nucleotide sequence in memory and return its location
    :param input_seq: Iterable representing a nucleotide sequence, may only contain the characters T,G,C,A
    :return: (offset, length) Tuple, where offset is the index from which this newly added sequence is stored
             and length is the number of subsequent indexes that are part of this sequence
    """
    if type(input_seq) != TwoBitArray:
        input_seq = TwoBitArray.from_iterable(input_seq, _nucleotide_syntax)
    _lock.acquire()
    offset = len(_seq)
    length = len(input_seq)
    _seq.extend(input_seq)
    _active.extend([1 for _ in range(len(input_seq))])
    _lock.release()
    return offset, length

def get_seq() -> TwoBitArray:
    """get a copy of the sequence"""
    return _seq.copy()

def prune_inactive() -> None:
    global _seq, _active
    _lock.acquire()
    _seq = _seq[_active]
    _active = bitarray(len(_seq))
    _active.setall(1)
    _lock.release()

def get(offset:int, length:int) -> TwoBitArray:
    return _seq[offset: offset + length]

def make_inactive(offset:int, length: int) -> None:
    # locking may be unnecessary here
    _lock.acquire()
    _active[offset:length] = 0
    _lock.release()

def bulk_make_inactive(offset:List[int], length:List[int]) -> None:
    _lock.acquire()
    for o, l in zip(offset, length):
        _active[o:l] = 0
    _lock.release()

def decode(offset:int, length:int) -> List[str]:
    return _seq.decode(offset, length)

def acquire_lock():
    """Acquire sequences lock, should only be used outside of this model when reindexing"""
    _lock.acquire()

def release_lock():
    _lock.release()

def is_complement(offset_a:int, offset_b:int, length:int) -> bool:
    """
    Check if the two sequences specified by offset and length are element-wise complementary
    :param offset_a: offset of first sequence
    :param offset_b: offset of second sequence
    :param length: length of both sequences (sequences of different lengths are by definition not element-wise complementary)
    :return: True if the two sequences are element-wise complementary, otherwise False.
    """
    return _seq[offset_a:length].is_complement(_seq[offset_b:length])

def get_dead_fraction() -> float:
    """Get fraction of strand entries which are dead"""
    return _active.count(0) / len(_active)