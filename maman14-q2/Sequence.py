from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numba.experimental import jitclass
from numba import njit, uint8

# Constants
# nucleotide encoding - complementarity expressed via bitwise-XOR
T = 0x01 # 00000001
G = 0x02 # 00000010
A = 0xFE # 11111110
C = 0xFD # 11111101
CHAR_TO_CODE = {"T":T, "G":G, "A":A, "C":C}
CODE_TO_CHAR = {T:"T", G:"G", A:"A", C:"C"}

@njit
def encode_from_str(input_str:str) -> NDArray[np.uint8]:
    """
    Encode a DNA sequence represented by a string of characters T,G,A and C into numpy vector
    :param input_str: String made up of ONLY the characters T,G,A,C
    :return: 1D uint8 numpy array
    """
    output = list()
    for c in input_str:
        value = CHAR_TO_CODE.get(c)
        if value is None: # c is not T,G,A or C
            raise ValueError("input string must be made up of the characters T,G,A,C only")
        output.append(value)
    return np.array(output, dtype=np.uint8)

@njit
def decode_nucleotide_sequence(seq:NDArray[np.uint8]) -> str:
    """
    Generate a string representation of a nucleotide sequence out of its encoded form
    :param seq: uint8 numpy array
    :return: string representing the sequence using the characters T,G,A and C
    """
    output = ""
    for entry in seq:
        value = CODE_TO_CHAR.get(entry)
        if value is None: # value is not an encoding for T,G,A or C
            raise ValueError("The value " + str(entry) + "is not a valid encoding")
        output += value
    return output

sequence_spec = [
    ('arr', uint8[:]) # numpy array encoding the sequence
]

# TODO: implement sequence class. consider how to express connected sequences