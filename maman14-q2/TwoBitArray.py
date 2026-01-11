from bitarray import bitarray
from typing import Iterable, Tuple, List

class TwoBitArray:
    """Array of 2-bit elements, implemented using bitarray.
    Supports a syntax which matches each of the 4 possible values to some custom value"""

    DEFAULT_SYNTAX = [0,1,2,3]
    ENDIAN = 'little'

    def __init__(self, syntax=DEFAULT_SYNTAX):
        """
        Initialize the TwoBitArray
        :param syntax: 4-items-long syntax list, matching by index to [00, 01, 10, 11]. defaults to [0,1,2,3].
        """
        self.left = bitarray()  # most significant bit
        self.right = bitarray() # least significant bit
        self.to_syntax = {(0,0): syntax[0],
                       (0,1): syntax[1],
                       (1,0): syntax[2],
                       (1,1): syntax[3]}
        self.from_syntax = {syntax[0]: (0,0),
                            syntax[1]: (0,1),
                            syntax[2]: (1,0),
                            syntax[3]: (1,1)}

    @classmethod
    def from_iterable(cls, seq: Iterable, syntax=DEFAULT_SYNTAX) -> TwoBitArray:
        """
        Initialize TwoBitArray from an iterable sequence.
        The iterable sequence must be made up only of values specified in the syntax.
        :param seq: iterable sequence representing two-bit array
        :param syntax: 4-item list matching by index to [00,01,10,11]. defaults to [0,1,2,3].
        :return: new TwoBitArray
        """
        seq = list(seq)
        obj = cls(syntax)
        # fill bitarrays
        for val in seq:
            try:
                binary_val = obj.from_syntax[val]
            except KeyError:
                raise ValueError("value " + val + " does not match syntax")
            obj.left.append(binary_val[0])
            obj.right.append(binary_val[1])
        return obj

    def append_int(self, val: int) -> None:
        """Append to array a 2-bit value represented as an int
        raises ValueError if val is more than 2 integers long or is made up of any integer other than 0 and 1"""
        if val / 100 != 0:
            raise ValueError("value " + val + " is too long. 1 or 2 bit value expected")
        most_sig, least_sig = divmod(val, 10)
        if (most_sig == 0 or most_sig == 1) and (least_sig == 0 or least_sig == 1):
            self.left.append(most_sig)
            self.right.append(least_sig)
        raise ValueError("cannot interpret " + str(val) + " as binary")

    def append(self, val) -> None:
        """
        Append to the array.
        The input value will first be interpreted as a syntax value, if this value is not in the syntax, it falls
        back on append_int, treating it as an explicit binary representation.
        :param val: Value to append
        """
        true_val = self.from_syntax.get(val)
        if true_val is None:
            if type(val) is not int:
                raise ValueError("Cannot interpret " + val)
            self.append_int(val)
            return
        self.left.append(true_val[0])
        self.right.append(true_val[1])

    # def extend(self, vals: Iterable) -> None:
    #     """Recursively append values to the array using the append function"""
    #     [self.append(item) for item in vals]

    def __len__(self):
        return len(self.left)

    def get_by_mask(self, mask:bitarray):
        if len(mask) != len(self.left):
            raise ValueError("mask is not of the same length as the array")
        output = TwoBitArray()
        output.to_syntax, output.from_syntax = self.to_syntax, self.from_syntax
        output.left, output.right = self.left[mask], self.right[mask]
        return output

    def __getitem__(self, key) -> Tuple[int, int]|TwoBitArray:
        if isinstance(key, slice):      # array slicing
            start, stop, step = key.indices(len(self))
            output = TwoBitArray()
            output.to_syntax, output.from_syntax = self.to_syntax, self.from_syntax
            output.left, output.right = self.left[start,stop,step], self.right[start,stop,step]
            return output
        if isinstance(key, bitarray):   # by mask
            return self.get_by_mask(key)
        return self.left[key], self.right[key]  # by index

    def get(self, key):
        """Return the index at that key in its syntax form"""
        val = self.left[key], self.right[key]
        return self.to_syntax[val]

    # TODO: to_list, maybe?

    def xor_mask(self, other: TwoBitArray) -> bitarray:
        """
        Return a bitarray mask where the element at index i is 1 iff the elements at index i of self and other are
        bitwise complementary, meaning the bitwise XOR operator returns 11.  otherwise the element is equal to 0.
        Both arrays must be of the same length.
        :param other: TwoBitArray to compare
        :return: bitarray -- index i represents if self[i] and other[i] are bitwise complementary
        """
        if len(self) != len(other):
            raise ValueError("length mismatch")
        left_xor = self.left ^ other.left       # XOR on most significant bit
        right_xor = self.right ^ other.right    # XOR on least significant bit
        return left_xor & right_xor             # output gets value 1 in any index where left_xor and right_xor are both 1

    def __xor__(self, other: TwoBitArray) -> bitarray:
        """Define (self ^ other) to be shorthand for self.xor_mask(other)"""
        return self.xor_mask(other)

    def copy(self):
        output = TwoBitArray()
        output.to_syntax, output.from_syntax = self.to_syntax, self.from_syntax
        output.left, output.right = self.left.copy(), self.right.copy()
        return output

    def decode(self, start:int, stop:int) -> List:
        output = [self.get(i) for i in range(start, stop)]
        return output

    def is_complement(self, other:TwoBitArray) -> bool:
        """
        Check if both arrays are element-wise complement to one another.
        :param other: TwoBitArray
        :return: True if for every index i, self[i] ^ other[i], otherwise False.
                 Arrays of different lengths will always return False.
        """
        try:
            return (self ^ other).all()
        except ValueError: # ValueError is raised on length mismatch
            return False

    def invert(self) -> TwoBitArray:
        """Get the bitwise complement by flipping every bit"""
        output = self.copy()
        output.left.invert(), output.right.invert()
        return output

    def __invert__(self) -> TwoBitArray:
        return self.invert()

    def search(self, sub_array:TwoBitArray):
        """
        Extension of bitarray's search function.
        Return iterator over indices where sub_array is found (starts)
        :param sub_array: Pattern being searched for
        """
        left_search = self.left.search(sub_array.left)
        right_search = self.right.search(sub_array.right)
        try:
            a, b = next(left_search), next(right_search)
            while True:
                if a == b:
                    yield a
                    a, b = next(left_search), next(right_search)
                elif a < b:
                    a = next(left_search)
                else:
                    b = next(right_search)
        except StopIteration:
            return

    def extend(self, other: TwoBitArray):
        """append the other TwoBitArray to the end of this TwoBitArray.
        self's syntax is retained and other's syntax has no influence on the extended array"""
        self.left.extend(other.left)
        self.right.extend(other.right)

    def merge(self, other: TwoBitArray) -> TwoBitArray:
        """Get a new TwoBitArray by appending other to the end of self.
        Identical to the 'extend' function, with the exception that merge returns a new
        TwoBitArray rather than modify self"""
        output = self.copy()
        output.extend(other)
        return output

    # TODO: convert to boolean NDArray, maybe?