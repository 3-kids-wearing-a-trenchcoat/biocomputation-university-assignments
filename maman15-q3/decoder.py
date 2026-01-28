from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict
import parse_sequence
import transcode
from DropletGenerator import IDX_DTYPE, BARCODE_BASES
from collections import Counter
from gauss_jordan_elimination import form_into_mat, gauss_jordan_elimination

# constants

# Some letters in the language are represented by a 50-50 ratio between two bases.
# If the two most frequent bases in a certain index of the sequences are within this margin of being 50-50,
# we consider it to be a double-base representation.
# That is, if the occurrence fraction of the two most frequent bases is in the range [0.5 - margin, 0.5 + margin].
# If a base's occurrence fraction is between [1 - margin, 1], we consider this to be a single-base represented letter.
# If neither, some terrible mass-data-corruption took place.
BASE_REP_MARGIN = 0.1
# BASE_REP_MARGIN_MAX = 1 - BASE_REP_MARGIN
BASE_REP_MARGIN_BELOW_HALF = 0.5 - BASE_REP_MARGIN
BASE_REP_MARGIN_ABOVE_HALF = 0.5 + BASE_REP_MARGIN
# clusters (by barcode) which have fewer than this many strands are discarded
MIN_IN_CLUSTER = 10

def _consensus_char(n: int, sequences: List[str]) -> Tuple[str, str]:
    """
    Get the consensus among the given sequences for the n-th character.
    :param n: index of character to examine
    :param sequences: input sequences
    :return: a tuple of characters representing bases as defined in transcode.LETTER_BASE_MAP.
             If a letter is represented by a single base, both elements in the output are the same.
             Otherwise, the tuple contains the two elements whose 50-50 ratio defines the letter.
    """
    base_in_n = [seq[n] for seq in sequences]
    counts = Counter(base_in_n)
    if len(counts) == 1:
        output = list(counts.keys())[0]
        return output, output
    candidate1, candidate2 = counts.most_common(2)  # get the two most common characters and their occurrence num
    total_occurrences = counts.total()
    base1, frac1 = candidate1[0], candidate1[1] / total_occurrences
    base2, frac2 = candidate2[0], candidate2[1] / total_occurrences
    # If the two most common bases are within margin of splitting 50-50, return both in lexicographic order
    if (BASE_REP_MARGIN_BELOW_HALF <= frac1 <= BASE_REP_MARGIN_ABOVE_HALF and
        BASE_REP_MARGIN_BELOW_HALF <= frac2 <= BASE_REP_MARGIN_ABOVE_HALF):
        return (base1, base2) if (base1 < base2) else (base2, base1)
    # Otherwise, return the most frequent occurring base
    return base1, base1

def _build_by_consensus(sequences) -> Tuple[str, str]:
    """
    Construct the DNA representation of the data by forming a consensus among all given strings.
    Iteratively, each position in all the given strings is examined and the most common character is chosen
    as the consensus character UNLESS the two most common characters are (within a margin) split 50-50,
    in which case another letter is represented.
    :param sequences: list of DNA sequences represented as strings
    :return: Two strings representing the consensus DNA representation of the data.
             For each index `n`, the `n`th character of the output will be the consensus at pos `n`.
             Meaning if there isn't a 50-50 split (within margin), both strings in output will contain the same
             character in position `n`.
             Otherwise, they will contain different letters at pos `n`, indicating a 50-50 split at that pos.
    """
    # Filter so that only sequences of the most common length are kept
    # (This filters out corruptions caused by missing or "extra" bases)
    target_len = max([len(s) for s in sequences])
    seqs = [seq for seq in sequences if len(seq) == target_len]
    # Build the two output strings, character by character, using _consensus_char
    out_list_a, out_list_b = [], []
    for i in range(target_len):
        a_addition, b_addition = _consensus_char(i, seqs)
        out_list_a.append(a_addition), out_list_b.append(b_addition)
    return "".join(out_list_a), "".join(out_list_b)

def sequence_droplet(sequences: List[str], min_in_cluster: int = MIN_IN_CLUSTER) -> List[NDArray[np.bool]]:
    """
    Generate the segments encoded in this droplet.
    :param sequences: Sequences to decode
    :param min_in_cluster: clusters that have fewer than this many members are discarded as noise.
                           defaults to MIN_IN_CLUSTER.
    :return: List of boolean numpy arrays, each is the binary representation of a segment.
             The barcode portion of each binary representation has been removed, as it is not needed
             beyond this process.
    """
    # Divide sequences into clusters by their barcode prefix. (
    clusters = parse_sequence.bucket_strings_by_prefix(sequences, BARCODE_BASES, True)
    output: List[NDArray[np.bool]] = []
    max_length = 0
    for bucket in clusters.values():
        if len(bucket) < min_in_cluster:
            continue    # ignore buckets below the minimum size
        strand1, strand2 = _build_by_consensus(bucket)  # recreate the two representative DNA strands via consensus
        try:
            in_language = transcode.from_DNA_to_words(strand1, strand2) # convert the two representative strands into the language
        except KeyError:
            # If key error encountered, it means we've got an un-mapped pair of bases in strand1 and strand2.
            # Meaning this bucket is corrupt beyond use, skip it and hope the other buckets can make up for it.
            continue
        if len(strand1) % 2 == 1 or len(strand2) % 2 == 1:
            # if strands are made of an odd number of words, it is necessarily missing some data
            # as every word in the language is made up of two letters.
            # In this case, this bucket is deemed too corrupt to use, and we skip it in the output
            continue
        try:
            output.append(transcode.from_words_to_np(in_language))  # convert language into binary
            if output[-1].size > max_length:
                max_length = output[-1].size
        except KeyError:
            continue
    # return output
    # prune all arrays that are not of the max size
    return [entry for entry in output if entry.size == max_length]

def decode_data(sequences: List[NDArray[np.bool]], segment_ids: List[NDArray[IDX_DTYPE]]) -> NDArray[np.bool]:
    """
    Decode data from list of sequences and their associated segments
    :param sequences: list of boolean numpy arrays representing a binary string.
                      Sequences should contain ONLY THE DATA PORTION, meaning no barcode or seed.
    :param segment_ids: for every `i`, segment_ids[i] contains a list of segment indexes that
                        the sequence sequences[i] encodes.
    :return: boolean numpy array representing the binary string that is the decoded data
    """
    A, Y = form_into_mat(segment_ids), form_into_mat(sequences)
    output = gauss_jordan_elimination(A, Y)
    return output.flatten()
