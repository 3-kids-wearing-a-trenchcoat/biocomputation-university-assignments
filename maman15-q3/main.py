from __future__ import annotations
from DropletGenerator import DropletGenerator
from typing import List
from decoder import sequence_droplet, decode_data
from parse_sequence import np_binary_to_str
from random import shuffle
from error_generator import inject_substitution_error, inject_deletion_error, inject_insertion_error

#constants
COPIES_PER_OLIGOMER = 100

# input -- The input sequence given in the assignment
EXAMPLE_SEQUENCE: str = ("000111111000101100001000000010000010111000010010100111100110000100000000000000110011000000110"
                         "000001100010010111001101010011100000110011100000000001011000111101001111011001110001101001111"
                         "111111111111111111111011100100100110000100001001101010110010100110000111001000001000010011011"
                         "010001010001100000110010111001110111000110110110110011000111000111110011010011000110100111011"
                         "101100010000010001100110010011100110001011000111001000101000011010101100110100010000010100101"
                         "100101100101001110110110001110001100010100100111001000001000011100101001101001000011100111010"
                         "100011110110000111001110010001010100010001101000110000100100100001001110011110011011010100011"
                         "011111001111111101111111101010111101110111101111111100111110000111101101011111110011111110101"
                         "011110000011111010011110111101110111110100111111111011101111101110111010010111100111111100111"
                         "111111111111001110100000110100111000100000000100010101011000001000100000000000111100001111000"
                         "000000100000001000011111111011000000101111111001001110000000001100110000000001110001111000010"
                         "010000100100001001000010100000100100011110000100100011111000101010001011000111001011111100100"
                         "110001000100000111000111010001010010010111000100110001001100010111000101000101001010000110100"
                         "000100010000010001010001010101110101011100000011001010101010101010101010010111101101000100110"
                         "110101111010111110011100001101000111101000101000101010101010101010100111100010110111001100110"
                         "010001101110011000010110010101101101001001010111111011001001010000011100010110011110111000010"
                         "110110011101010011010010110010101110001001101111000001110111011100001011100100010111010011111"
                         "110110011110011110000111100111000101000000010111111000111110000000011110010110111001100110111"
                         "111001011110011111111011101111000000011111110011111111100001111110011100001000001011110001111"
                         "110111111100000001010000011010000010100011110000101100011110111000000010110001101001001110000"
                         "000001000111011110000111100001111001100011111111000011110001")

# functions
def encode(generator: DropletGenerator, copies_per_oligomer: int = COPIES_PER_OLIGOMER) -> List[str]:
    """
    Encode data sequence as droplets, multiply them and shuffle.
    :param generator: DropletGenerator
    :param copies_per_oligomer: How many copies of each generated oligomer should be in the final output
    :return: List of strings representing DNA sequences
    """
    oligomers = generator.bulk_gen_as_DNA() * copies_per_oligomer
    shuffle(oligomers)
    return oligomers

def run_experiment(input_seq: str = EXAMPLE_SEQUENCE, seed_length: int|None = None,
                   sub_error_prob: float = 0, del_error_prob: float = 0, insert_error_prob: float = 0,
                   print_messages: bool = True,) -> str:
    """
    Run a round-trip, DNA-storage experiment on the input data.
    Encode the input sequence into droplets, generate strands from the droplets, multiply the strands,
    shuffle them and then decode the message.
    :param input_seq: Data to be encoded into DNA, given as a string made up only of characters '0' and '1'
    :param seed_length: Maximum length of the binary representation of the seed.
                        If the calculated needed seed length is smaller, that smaller value will be used.
    :param sub_error_prob: probability of each base in each oligomer to be substituted for the wrong base.
                           Expected values in range [0,1]. defaults to 0.
    :param del_error_prob: probability of each base in each oligomer to be deleted.
                           Expected values in range [0,1]. defaults to 0.
    :param insert_error_prob: probability of each space in an oligomer (between bases, before first base
                              or after last base) to have a base added to it.
                              Expected values in range [0,1]. defaults to 0.
    :param print_messages: Whether to print messages
    :return: The decoded message in the form of a string of characters '0' and '1'
    """
    # encode data into droplets
    bits_per_word = 5   # Hard-coded here as the transcode module is implemented specifically for a 5-bit word
    droplet_generator = DropletGenerator(input_seq, bits_per_word, seed_length)
    oligomers = encode(droplet_generator)
    # inject errors
    replacement_errors = inject_substitution_error(oligomers, sub_error_prob)
    if print_messages:
        print("Replacement errors injected: " + str(replacement_errors))
    deletion_errors = inject_deletion_error(oligomers, del_error_prob)
    if print_messages:
        print("Deletion errors injected: " + str(deletion_errors))
    insertion_errors = inject_insertion_error(oligomers, insert_error_prob)
    if print_messages:
        print("Insertion errors injected: " + str(insertion_errors))
        print("Total errors: " + str(replacement_errors + deletion_errors + insertion_errors))
    # decode
    sequenced_oligomers = sequence_droplet(oligomers)
    segment_idxs = [droplet_generator.find_segments(entry) for entry in sequenced_oligomers]
    data = [seq[droplet_generator.seed_binary_length:] for seq in sequenced_oligomers]
    decoded_seq_bin = decode_data(data, segment_idxs)
    decoded_seq =  np_binary_to_str(decoded_seq_bin)
    # print success/failure
    if print_messages:
        if decoded_seq == input_seq:
            print("input sequence and output sequence are identical!")
        else:
            print("SEQUENCE MISMATCH!!")
            print("input:")
            print(input_seq)
            print("output:")
            print(decoded_seq)
    return decoded_seq

if __name__ == "__main__":
    print(run_experiment(seed_length=55, print_messages=True, sub_error_prob= 1e-5,
                         del_error_prob=1e-5, insert_error_prob=1e-5))