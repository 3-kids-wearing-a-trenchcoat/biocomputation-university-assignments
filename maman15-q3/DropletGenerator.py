from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import parse_sequence
import transcode
from math import comb, log2, ceil
import itertools

# constants
POSSIBLE_RANKS = [2, 13]  # a droplet randomly chooses one of these two rank values with equal probability
IDX_DTYPE = np.uint64
MASTER_SEED = 43523345234
BARCODE_BASES = 10        # number of bases that make up each barcode, a base is effectively equivalent to 2-bits
BARCODE_BITS = ceil((BARCODE_BASES / 2) * 5)
BARCODE_UPPER_BOUND = 2 ** BARCODE_BITS    # Max decimal value of a barcode
BULK_GENERATION_OVERHEAD = 0.3
DISALLOWED_LETTERS_IN_BARCODE = {"Y", "Z"}   # limit barcode letters to those that are not 50-50 rep in DNA
ALLOWED_BARCODE_BIN = [entry[1] for entry in transcode.WORD_BIT_PAIRS
                       if not any(forbidden in entry[0] for forbidden in DISALLOWED_LETTERS_IN_BARCODE)]

def calc_number_of_seeds(segment_num: int) -> Tuple[int, int]:
    """
    Calculate the number of possible seeds a droplet can choose as a function of the number of segments
    :param segment_num: number of segments the input sequence was divided to
    :return: Tuple containing these two ints in order:
             1. Number of possible seeds that a droplet should choose
             2. Number of bits needed to represent these values
    """
    # TODO: divisor should be specified rather than baked in as 5
    seg_permutations = sum([comb(segment_num, rank) for rank in POSSIBLE_RANKS])
    binary_rep_length = ceil(log2(seg_permutations))
    # round binary_rep_length up to nearest multiple of 5, to keep resulting droplet sequences of length
    # divisible by 5.
    binary_rep_length += (5 - binary_rep_length) % 5
    return seg_permutations, binary_rep_length


class DropletGenerator:
    # TODO: (extended?) Hamming code error correction
    def __init__(self, input_seq: str, bits_per_word: int = 5):
        """
        Initialize the droplet generator.
        Split the input sequence into segments, convert each to a boolean numpy array and initialize the static
        list of segments from which each droplet will choose at random.
        :param input_seq: string made up only of the characters '0' and '1', representing the binary input sequence
        :param bits_per_word: number of bits used to represent a word in the language
        """
        # TODO: seed length can probably be smaller and explicitly defined as input
        if len(input_seq) % bits_per_word != 0:
            raise ValueError("length of the input sequence is not divisible by bits_per_word")
        self.rng = np.random.default_rng(MASTER_SEED)
        # split input sequence (str) into unique, equal-length sequences (np.bool)
        str_segments = parse_sequence.split_into_unique_segments(input_seq, bits_per_word)
        self.segments = parse_sequence.convert_segments_to_bool_ndarrays(str_segments)
        self.seg_idx = np.arange(len(self.segments)).astype(IDX_DTYPE)
        # Calculate number of seeds needed and choose that many seeds randomly
        self.seed_num, self.seed_binary_length = calc_number_of_seeds(len(self.segments))
        self.used_barcode_values = set()
        # I am working with the assumption that seed_binary_length, BARCODE_BITS
        # and the sequence length are all divisible by bits_per_word, meaning so is the concatenated droplet sequence
        # TODO: SANITY CHECK
        self.seeds_made_in_order = []
        self.bin_seeds_made_in_order = []

    # ===== ENCODING PHASE =====

    # def check_barcode(self, barcode: int) -> bool:
    def check_barcode(self, barcode: NDArray[np.bool]) -> bool:
        """
        Check if the given barcode has been used before. If not, add it to the set of used barcodes
        :param barcode: binary value of barcode represented by a boolean numpy array
        :return: 'True' if this barcode wasn't found in the used set, otherwise 'False'.
                 (More accurately; 'True' means this barcode has JUST been added to the used barcode values)
        """
        barcode_tup = tuple(barcode.tolist())
        if barcode_tup in self.used_barcode_values:
            return False
        self.used_barcode_values.add(barcode_tup)
        return True

    @staticmethod
    def gen_barcode(rng) -> NDArray[np.bool]:
        """Generate a barcode made up only of letters that are not expressed by
        a 50-50 ratio of bases in its DNA representation"""
        return rng.choice(ALLOWED_BARCODE_BIN, int(BARCODE_BASES / 2)).flatten()

    def gen_droplet(self) -> NDArray[np.bool]:
        """
        Generate a binary sequence representing a droplet.
        This droplet is used as a "blueprint" for "true" droplets, which are genetically coded and contain multiples
        :return: boolean NDArray representing the binary sequence
        """
        # Choose seed for droplet
        # seed = self.rng.integers(0, self.seed_num).astype(IDX_DTYPE)
        seed = int(self.rng.integers(0, self.seed_num))
        seed_bin = parse_sequence.uint_to_binary(seed, self.seed_binary_length)

        # TODO: sanity check
        self.seeds_made_in_order.append(seed)

        droplet_rng = np.random.default_rng(seed)
        # randomly choose INDEX OF rank and then choose SEGMENT INDEXES accordingly
        # THE ORDER IS IMPORTANT as we'll rely on it for decoding
        rank_idx = droplet_rng.integers(0, len(POSSIBLE_RANKS))
        rank = POSSIBLE_RANKS[rank_idx]
        segments_idx = droplet_rng.choice(self.seg_idx, rank, replace=False)

        # Generate a unique barcode
        # barcode = droplet_rng.choice([True, False], BARCODE_BITS)
        barcode = self.gen_barcode(droplet_rng)
        while not self.check_barcode(barcode):
            # barcode = droplet_rng.choice([True, False], BARCODE_BITS)
            barcode = self.gen_barcode(droplet_rng)

        # calculate payload -- the portion of the droplet's sequence that actually contains the data
        payload = np.bitwise_xor.reduce([self.segments[i] for i in segments_idx])
        # droplet sequence as a concatenation (in order) of barcode_bin, seed_bin and payload
        sequence = np.concatenate((barcode, seed_bin, payload))
        # Append zeros to the end of the sequence if necessary
        return sequence

    def bulk_gen_droplets(self, in_language: bool = False, n: int|None = None) -> List[NDArray[np.bool]] | List[str]:
        """
        Generate several droplets at once using gen_droplet.
        :param n: Number of droplets to generate.
                  If 'None'; generates ceil[1.05 * num_of_segments] droplets
        :param in_language: Whether to return the output in the language
        :return: List of boolean numpy arrays representing sequences if in_language is 'False'.
                 if in_language is 'True', return a list of strings representing sequences.
        """
        if n is None:
            n = ceil((1 + BULK_GENERATION_OVERHEAD) * len(self.segments))
        output = [self.gen_droplet() for _ in range(n)]
        if in_language:
            return [transcode.from_np_to_words(entry) for entry in output]
        return output

    def bulk_gen_as_DNA(self, n: int|None = None) -> List[str]:
        """
        Generate several droplets at once using gen_droplet
        :param n:
        :return:
        """
        droplets: List[str] = self.bulk_gen_droplets(True, n)
        DNA_pairs = [transcode.from_words_to_DNA(entry) for entry in droplets]
        output = list(itertools.chain.from_iterable(DNA_pairs))
        return output

    def find_segments(self, sequence: NDArray[np.bool]) -> NDArray[IDX_DTYPE]:
        """
        Find the segments the given sequence refers to via the encoded seed value.
        :param sequence: encoded sequence WITHOUT THE BARCODE
        :return: List of integers representing segment indexes
        """
        # find seed
        seed_binary = sequence[:self.seed_binary_length]
        seed = parse_sequence.binary_to_uint(seed_binary)
        # find segments based on seed
        droplet_rng = np.random.default_rng(seed)
        # randomly choose INDEX OF rank and then choose SEGMENT INDEXES accordingly
        # THE ORDER IS IMPORTANT as we'll rely on it for decoding
        rank_idx = droplet_rng.integers(0, len(POSSIBLE_RANKS))
        rank = POSSIBLE_RANKS[rank_idx]
        segments_idx = droplet_rng.choice(self.seg_idx, rank, replace=False)
        return segments_idx
