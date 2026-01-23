from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import parse_sequence
import transcode
from math import ceil

# constants
POSSIBLE_RANKS = [2, 13]  # a droplet randomly chooses one of these two rank values with equal probability
IDX_DTYPE = np.uint16
MASTER_SEED = 43523345234
MAX_SEED_VAL = 999999
BARCODE_BASES = 10        # number of bases that make up each barcode, a base is effectively equivalent to 2-bits
BARCODE_BITS = BARCODE_BASES * 2
BARCODE_UPPER_BOUND = 2 ** BARCODE_BITS    # Max decimal value of a barcode
BULK_GENERATION_OVERHEAD = 0.05

class DropletGenerator:
    def __init__(self, input_seq: str, bits_per_word: int = 5):
        """
        Initialize the droplet generator.
        Split the input sequence into segments, convert each to a boolean numpy array and initialize the static
        list of segments from which each droplet will choose at random.
        :param input_seq: string made up only of the characters '0' and '1', representing the binary input sequence
        :param bits_per_word: number of bits used to represent a word in the language
        """
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
        # Calculate appended bits needed for each rank.
        # appended_bits_num[i] is the number of junk bits appended to the end of the sequence of a droplet of rank POSSIBLE_RANKS[i]
        # a sequence is made up of a barcode (BARCODE_BITS), a seed (self.seed_binary_length)
        # and a payload (bits for a single payload times the rank)
        seg_len = len(self.segments[0])
        sequence_length = [BARCODE_BITS + self.seed_binary_length + (seg_len * rank) for rank in POSSIBLE_RANKS]
        # TODO: I AM ASSUMING I NEED TO EXTEND IT TO BE A COMPLETE WORD, I.E A MULTIPLE OF 5.
        # This may be wrong, it may also be in conflict with how I split the input seq, check into it if something doesn't work
        self.appended_bits_num = [seq_len % bits_per_word for seq_len in sequence_length]

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
        # randomly choose INDEX OF rank and then choose SEGMENT INDEXES accordingly
        # THE ORDER IS IMPORTANT as we'll rely on it for decoding
        rank_idx = droplet_rng.integers(0, len(POSSIBLE_RANKS))
        rank = POSSIBLE_RANKS[rank_idx]
        # rank = droplet_rng.choice(POSSIBLE_RANKS)
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
        # droplet sequence as a concatenation (in order) of barcode_bin, seed_bin and payload
        sequence = np.concatenate((barcode_bin, seed_bin, payload), dtype=np.bool)
        # Append zeros to the end of the sequence if necessary
        if self.appended_bits_num[rank_idx] == 0:
            return sequence
        return np.concatenate((sequence, np.zeros(self.appended_bits_num[rank_idx], dtype=np.bool)))

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
        # return [transcode.from_words_to_DNA(entry) for entry in droplets]
        DNA_pairs = [transcode.from_words_to_DNA(entry) for entry in droplets]
        return sum(DNA_pairs, [])   # flatten list


    # TODO: decode words into binary sequence

    # TODO: decode DNA into binary via words

