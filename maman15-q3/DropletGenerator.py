from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import parse_sequence

# constants
POSSIBLE_RANKS = [2, 13]  # a droplet randomly chooses one of these two rank values with equal probability
IDX_DTYPE = np.uint16
MASTER_SEED = 43523345234
MAX_SEED_VAL = 999999
BARCODE_BASES = 10        # number of bases that make up each barcode, a base is effectively equivalent to 2-bits
BARCODE_BITS = BARCODE_BASES * 2
BARCODE_UPPER_BOUND = 2 ** BARCODE_BITS    # Max decimal value of a barcode

class DropletGenerator:
    def __init__(self, input_seq: str, bits_per_word: int = 5):
        """
        Initialize the droplet generator.
        Split the input sequence into segments, convert each to a boolean numpy array and initialize the static
        list of segments from which each droplet will choose at random.
        :param input_seq: string made up only of the characters '0' and '1', representing the binary input sequence
        :param bits_per_word: number of bits used to represent a word in the language
        """
        # TODO: I'm assuming the input is divisible by 5, should probably think of what to do if it isn't.
        self.rng = np.random.default_rng(MASTER_SEED)
        # split input sequence (str) into unique, equal-length sequences (np.bool)
        str_segments = parse_sequence.split_into_unique_segments(input_seq, bits_per_word)
        self.segments = parse_sequence.convert_segments_to_bool_ndarrays(str_segments)
        self.seg_idx = np.arange(len(self.segments), dtype=IDX_DTYPE)
        # Calculate number of seeds needed and choose that many seeds randomly
        self.seed_num, self.seed_binary_length = parse_sequence.calc_number_of_seeds(len(self.segments))
        # self.possible_seeds = self.rng.integers(MAX_SEED_VAL, size=seed_num, dtype=np.uint32)
        # self.possible_seeds = np.arange(MAX_SEED_VAL, dtype=IDX_DTYPE)
        self.used_barcode_values = set()

    def check_barcode(self, barcode: int) -> bool:
        """
        Check if the given barcode has been used before. If not, add it to the set of used barcodes
        :param barcode: decimal representation of desired barcode value, must be a non-negative integer
        :return: 'True' if this barcode wasn't found in the used set, otherwise 'False'.
                 (More accurately; 'True' means this barcode has JUST been added to the used barcode values)
        """
        if barcode < 0:
            raise ValueError("barcode decimal representation must be a non-negative integer")
        if barcode in self.used_barcode_values:
            return False
        self.used_barcode_values.add(barcode)
        return True

    def gen_droplet(self) -> NDArray[np.bool]:
        """
        Generate a binary sequence representing a droplet.
        This droplet is used as a "blueprint" for "true" droplets, which are genetically coded and contain multiples
        :return: boolean NDArray representing the binary sequence
        """
        # Choose seed for droplet
        seed = self.rng.integers(0, self.seed_num, dtype=IDX_DTYPE)
        droplet_rng = np.random.default_rng(seed)
        # randomly choose rank and then choose segments accordingly
        # THE ORDER IS IMPORTANT as we'll rely on it for decoding
        rank = droplet_rng.choice(POSSIBLE_RANKS)
        segments_idx = droplet_rng.choice(self.seg_idx, rank, replace=False)
        # Generate a unique barcode
        barcode = droplet_rng.integers(0, BARCODE_UPPER_BOUND)
        while True:
            if self.check_barcode(barcode):
                break
            barcode = droplet_rng.integers(0, BARCODE_UPPER_BOUND)
        # Get binary representation for barcode and seed
        barcode_bin = parse_sequence.uint_to_binary(barcode, BARCODE_BITS)  # binary representation of barcode
        seed_bin = parse_sequence.uint_to_binary(seed, self.seed_num)
        # calculate payload -- the portion of the droplet's sequence that actually contains the data
        payload = np.bitwise_xor.reduce(self.segments[i] for i in segments_idx)
        # return droplet sequence as a concatenation (in order) of barcode_bin, seed_bin and payload
        return np.concatenate((barcode_bin, seed_bin, payload), dtype=np.bool)


