"""This module contains the functions that convert between any of the following
1. binary representation
2. language representation (two-letter words in a 6-letter language)
3. DNA representation"""

from __future__ import annotations
from typing import List, Tuple, TypeAlias
import numpy as np
from numpy.typing import NDArray

# representations
five_bit: TypeAlias = tuple[int, int, int, int, int]
WORD_BIT_PAIRS: List[Tuple[str, five_bit]] = [("UU", (0,0,0,0,0)), ("UV", (0,0,0,0,1)), ("UW", (0,0,0,1,0)),
                                              ("UX", (0,0,0,1,1)), ("UY", (0,0,1,0,0)), ("UZ", (0,0,1,0,1)),
                                              ("VU", (0,0,1,1,0)), ("VV", (0,0,1,1,1)), ("VW", (0,1,0,0,0)),
                                              ("VX", (0,1,0,0,1)), ("VY", (0,1,0,1,0)), ("VZ", (0,1,0,1,1)),
                                              ("WU", (0,1,1,0,0)), ("WV", (0,1,1,0,1)), ("WW", (0,1,1,1,0)),
                                              ("WX", (0,1,1,1,1)), ("WY", (1,0,0,0,0)), ("WZ", (1,0,0,0,1)),
                                              ("XU", (1,0,0,1,0)), ("XV", (1,0,0,1,1)), ("XW", (1,0,1,0,0)),
                                              ("XX", (1,0,1,0,1)), ("XY", (1,0,1,1,0)), ("XZ", (1,0,1,1,1)),
                                              ("YU", (1,1,0,0,0)), ("YV", (1,1,0,0,1)), ("YW", (1,1,0,1,0)),
                                              ("YX", (1,1,0,1,1)), ("YY", (1,1,1,0,0)), ("YZ", (1,1,1,0,1)),
                                              ("ZU", (1,1,1,1,0)), ("ZV", (1,1,1,1,1))]

# Matching a letter in the language to its representation in DNA bases.
# If the two elements in the sub-tuple are identical, that letter is represented by that letter directly.
# Otherwise, that letter is represented by a 50-50 distribution among the two bases in the sub-tuple
# For example, The letter "U" is represented by the base "A", and the letter "Y" is represented by a 50-50 distribution
# of the bases "A" and "G".
LETTER_BASE_MAP: List[Tuple[str, Tuple[str, str]]] = [("U", ("A", "A")), ("V", ("C", "C")),
                                                      ("W", ("G", "G")), ("X", ("T", "T")),
                                                      ("Y", ("A", "G")), ("Z", ("C", "T"))]

# dictionaries - matching between representations
WORD_TO_BITS = {entry[0]: entry[1] for entry in WORD_BIT_PAIRS}
BITS_TO_WORD = {entry[1]: entry[0] for entry in WORD_BIT_PAIRS}
LETTER_TO_BASE_PAIR = {entry[0]: entry[1] for entry in LETTER_BASE_MAP}
BASE_PAIR_TO_LETTER = {entry[1]: entry[0] for entry in LETTER_BASE_MAP}


# functions
def from_np_to_words(seq: NDArray[np.bool], word_len: int = 5) -> str:
    """
    Translate a binary sequence into the words it represents.
    :param seq: Binary sequence in the form of a boolean numpy array
    :param word_len: length of word in bits
    :return: a string of the decoded word
    """
    substrings: List[str] = []
    # assuming seq's length is divisible by word_len
    for start in range(0, seq.size, word_len):
        # also assuming all values I might encounter here are defined in the dict
        as_tuple = tuple(seq[start: start + word_len])
        substrings.append(BITS_TO_WORD[as_tuple])
    return "".join(substrings)

def from_words_to_np(seq: str, letters_in_word: int = 2) -> NDArray[np.bool]:
    """
    Generate the binary representation of the input phrase.
    :param seq: input phrase given as a string
    :param letters_in_word: Number of letters in a word, all words are assumed to be made up of exactly these many letters.
    :return: Boolean numpy array representing the binary sequence encoding the phrase.
    """
    bin_words = [np.array(WORD_TO_BITS[seq[i: i + letters_in_word]], dtype=np.bool) for i in range(0, len(seq), 2)]
    return np.concatenate(bin_words)

def from_words_to_DNA(seq: str) -> Tuple[str, str]:
    """
    Convert a language sequence into its DNA representation
    :param seq: sequence in language form
    :return: Two strings representing two DNA strands encoding the sequence.
             Two sequences are returned as some of the letters in the language are defined by a 50-50 ratio between bases.
    """
    out_arr1: List[str] = []
    out_arr2: List[str] = []
    for letter in seq:
        base_pair = LETTER_TO_BASE_PAIR[letter]
        out_arr1.append(base_pair[0]), out_arr2.append(base_pair[1])
    return "".join(out_arr1), "".join(out_arr2)

def _get_letter_by_base_pair(base1: str, base2: str) -> str:
    """Get the letter represented by this pair of parallel bases"""
    if len(base1) != 1 or len(base2) != 1:
        raise ValueError("Expected single character strings")
    pair = (base1, base2) if base1 <= base2 else (base2, base1)
    return BASE_PAIR_TO_LETTER[pair]

def from_DNA_to_words(strand1: str, strand2: str) -> str:
    """
    Covert a given DNA representation into a language sequence.
    The given DNA representation is represented by two strands that may be different.
    That is because certain letters (Y and Z) in the language are represented by a 50-50 ratio of different
    bases in DNA.
    """
    out_arr: List[str] = [_get_letter_by_base_pair(base1, base2) for base1, base2 in zip(strand1, strand2)]
    return "".join(out_arr)
