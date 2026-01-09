"""
This module is an abstraction for the list of all DNA sequences
All DNA sequences are stored in a packed (sequential) manner in a TwoBitArray.
Each individual strand will be represented by the tuple (ID, offset, length, active) and will use the functions
defined here to find the range nucleotides that actually represents it.
"""

from __future__ import annotations
from TwoBitArray import TwoBitArray
from bitarray import bitarray
from typing import Iterable, Tuple

_nucleotide_syntax = ['T', 'G', 'C', 'A'] # 'T':=00, 'G':=01, 'C':=10, 'A':=11
_seq = TwoBitArray(_nucleotide_syntax)    # The logical array in which all DNA sequences are stored
_active = bitarray()                      # bitmap representing whether the sequence is deleted (equals 0)

def seq_append(input_seq: str) -> Tuple[int, int]:
    """
    Store a nucleotide sequence in memory and return its location
    :param input_seq: string representing a nucleotide sequence, may only contain the characters T,G,C,A
    :return: (offset, length) Tuple, where offset is the index from which this newly added sequence is stored
             and length is the number of subsequent indexes that are part of this sequence
    """
    offset = len(_seq)
    length = len(input_seq)
    _seq.extend(input_seq)
    return offset, length

def get_seq() -> TwoBitArray:
    """get a copy of the sequence"""
    return _seq.copy()

def prune_inactive() -> None:
    global _seq, _active
    _seq = _seq[_active]
    _active = bitarray(len(_seq))
    _active.setall(1)

def get(offset:int, length:int) -> TwoBitArray:
    return _seq[offset:length]
