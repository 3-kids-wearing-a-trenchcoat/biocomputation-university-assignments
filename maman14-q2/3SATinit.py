"""Initialize simulation for a SAT3 problem"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import sequences, strand, binding
from typing import List, Tuple
from TwoBitArray import TwoBitArray
from tqdm import tqdm, trange
import math

# constants
NUCLEOTIDE_SYNTAX = ['T', 'G', 'C', 'A']
T = 1000000             # Number of complete assemblies (solutions) I want to simulate
SAFETY_FACTOR = 2       # Safety factor to account for inefficiencies
MAX_COMP_ALPHA = 0.3    # Maximum ratio of complementary between any two representations

# global vars
# simulation settings
rep_len: int|None = None                        # length of representation in nucleotides
max_complementarity: int|None = None            # maximum allowed complementarity between every two representations
copies_per_literal: int = math.ceil((T / 2) * SAFETY_FACTOR)     # Number of copies per literal (x_i) to start with
copies_per_connector: int = T * SAFETY_FACTOR   # number of copies of connectors (c_i) to start with
#  component representations
variable_rep_true: List[TwoBitArray] = []       # cell `i` holds the representation of assigning x_i the value 'True'
variable_rep_false: List[TwoBitArray] = []      # cell `i` holds the representation of assigning x_i the value 'False'
connector_rep: List[TwoBitArray] = []           # cell `i` holds the representation of the connector c_i
# constraint representations
constraint_rep: List[List[TwoBitArray]] = []    # cell `i` represents a boolean clause made up of 3 constraints
# each constraint is a concatenation of a literal (true of false variable) and a connector (representing position)

# functions
# ===== generate representations for literals and connectors =====
def dna_rep_params(n:int) -> Tuple[int, int]:
    """
    Get the representation length and max complementarity as a function of the number of unique variables.
    Length and max overlap are chosen in a way that makes random, unintentional bindings between components rare.
    :param n: number of unique variables
    :return: Tuple of 2 ints -- in order they are the representation length and max allowed complementary matches
    """
    l = math.ceil(12 + 4 * math.log2(n + 1))    # length of representation in nucleotides
    k = math.ceil(MAX_COMP_ALPHA * l)           # maximum complementary pairs between every two representations
    return l, k

def set_dna_rep_params(n:int) -> None:
    global rep_len, max_complementarity
    rep_len, max_complementarity = dna_rep_params(n)

def _existing_rep_generator():
    """Generator yielding existing representation for true literal, false literal or connector"""
    yield from variable_rep_true
    yield from variable_rep_false
    yield from connector_rep

def is_unique_enough(candidate:TwoBitArray) -> bool:
    """
    Check if a candidate representation is unique enough by making sure its complementarity with every other element
    generated so far is below the threshold.
    :param candidate: representation in TwoBitArray
    :return: 'True' if complementarity between the candidate and every existing representation is below threshold.
             'False' if at least one representation exists whose complementarity with the candidate is above threshold.
    """
    for existing in _existing_rep_generator():
        if (candidate ^ existing).count() > max_complementarity:
            return False
    return True

def get_new_representation() -> TwoBitArray:
    """Find a new, unique-enough representation"""
    candidate = TwoBitArray.random(rep_len, NUCLEOTIDE_SYNTAX)
    while not is_unique_enough(candidate):
        candidate = TwoBitArray.random(rep_len, NUCLEOTIDE_SYNTAX)
    return candidate

def generate_unique_representations(n: int) -> None:
    """
    Initialize the representation lists by generating a unique representation for every literal and every connector
    :param n: number of variables (n variables - n positive literals - n negative literals - n+1 connectors)
    """
    for i in range(n):  # generate n representations each for pos-literal, neg-literal and connectors
        variable_rep_true.append(get_new_representation())
        variable_rep_false.append(get_new_representation())
        connector_rep.append(get_new_representation())
    # generate one extra representation for connectors
    connector_rep.append(get_new_representation())

# ===== parse input =====
type Literal = Tuple[int, bool]                 # (variable number, literal is True)
type Clause = Tuple[Literal, Literal, Literal]  # (Clause[0] OR Clause[1] OR Clause[2])
type Formula = List[Clause]                     # Formula[0] AND Formula[1] AND ...

def validate_formula(formula:Formula) -> int:
    """
    Validate a 3SAT formula represented as:
      formula = list of clauses
      clause  = tuple of 3 literals
      literal = (var_index: int, polarity: +1 or -1)
    Raises a ValueError or TypeError if input is not valid
    :param formula: Formula as specified above
    :return: Number of literals
    """
    if not isinstance(formula, (list, tuple)):
        raise TypeError("Formula must be a list or tuple of clauses")
    if not formula:
        raise ValueError("Formula must contain at least one clause")

    seen_vars = set()
    for ci, clause in enumerate(formula):
        if not isinstance(clause, tuple):
            raise TypeError(f"Clause {ci} is not a tuple")
        if len(clause) != 3:
            raise ValueError(f"Clause {ci} does not contain exactly 3 literals")
        for li, literal in enumerate(clause):
            if not isinstance(literal, tuple):
                raise TypeError(f"Literal {li} in clause {ci} is not a tuple")
            if len(literal) != 2:
                raise ValueError(f"Literal {li} in clause {ci} must be (var, polarity)")

            var, pol = literal
            if not isinstance(var, int) or var < 0:
                raise ValueError(f"Invalid variable index {var} in clause {ci}")
            if pol not in (+1, -1):
                raise ValueError(f"Invalid polarity {pol} in clause {ci}")
            seen_vars.add(var)
    # Check consecutiveness
    max_var = max(seen_vars)
    expected = set(range(max_var + 1))
    if seen_vars != expected:
        missing = expected - seen_vars
        raise ValueError(f"Variable indices are not consecutive; missing {sorted(missing)}")
    return max_var + 1


# ===== generate constraints =====
def generate_constraints(formula: Formula) -> None:
    for clause in formula:  # for each clause
        constraints = []
        for literal in clause:
            # choose literal representation based on if the literal is the variable of its negation
            var_num = literal[0]
            literal_rep = variable_rep_true[var_num] if literal[1] else variable_rep_false[var_num]
            right_connector = connector_rep[var_num + 1]
            constraint = literal_rep.concat(right_connector)
            if var_num == 0:                # constraints for x_0 include connector c_0 as well as c_1
                constraint = connector_rep[0].concat(constraint)
            constraints.append(~constraint) # The complementary of the constraint we've built is what will bind to the desired sequence
        constraint_rep.append(constraints)

# ===== initialize 3SAT =====
def init_3sat(formula: Formula) -> None:
    """
    Initialize the 3SAT problem according to the formula.
    It is assumed the input formula is in 3SAT form, meaning clauses are all in an AND relation literals
    in each clause are in an OR relation.
    :param formula: List of clauses. Each Clause is a length 3 tuple of literals.
                    Each literal is an (int, bool) tuple representing the variable number and whether the literal
                    for that variable is true of false respectively.
    """
    n = validate_formula(formula)
    set_dna_rep_params(n)
    generate_unique_representations(n)
    generate_constraints(formula)
