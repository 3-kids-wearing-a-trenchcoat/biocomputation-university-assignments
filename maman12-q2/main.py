from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd
from pathlib import Path
from tqdm import tqdm, trange
from Individual import Individual, FTYPE
from Population import Population
from RNASeqDeconvolution import RNASeqDeconvolution

# default values
DEFAULT_PARAMS = {"rng": np.random.default_rng(123),
                  "mut_prob": 0.2,
                  "mut_standard_deviation": 0.2,
                  "crossover_prob": 0.9,
                  "max_iter": 2000,
                  # "max_iter": 999999999,
                  "satisfactory": 1e-3,
                  "stagnation_limit": 200,
                  # "stagnation_limit": 99999,
                  "stagnation_diff": 1e-3,
                  "pop_size": 200,
                  "win_prob": 0.7,
                  "init_sigma": 0.7,
                  "tournament_participants": 3,
                  "carry_over": 2,
                  "H_path": Path(__file__).parent / "matrices" / "gene_celltype_TPM.tsv",
                  "M_path": Path(__file__).parent / "matrices" / "gene_sample_TPM.tsv",
                  "true_result_path": Path(__file__).parent / "matrices" / "sample_celltype_groundT.tsv"
                  }
RESULT_LABELS = ["fitness score", "iterations", "cause of stop"]

def parse_input_matrix(path:Path, t:DTypeLike = FTYPE) -> NDArray:
    """
    turn the matrix in the given path into a numpy matrix, discarding row/column names
    :param path: path to file (using pathlib)
    :param t: dtype of output matrix (defaults to FTYPE as defined in Individual)
    :return: NDArray of type t
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    output = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=t)
    return output

def set_parameters(input_params:dict[str, float|int|NDArray[FTYPE]|np.random.Generator]) -> RNASeqDeconvolution:
    """
    Helper function which initializes the **many** variables across Population and Individual.
    Each variable-value pair in params has that value set for that variable.
    Every variable not in params is set to the default
    :param input_params: dict[str, Any] - key is variable name as a string, value is the value to assign to the variable
    :return: RNASeqDeconvolution object with the variables set as specified
    """
    # start with parameters as specified in defaults
    p = DEFAULT_PARAMS.copy()
    # change/add all specified key-value pairs in input_params
    for key, value in input_params.items():
        p[key] = value

    # set static Individual parameters
    Individual.set_static_vars(p["rng"], p["mut_prob"], p["mut_standard_deviation"],
                               p["crossover_prob"], p["M"], p["H"])
    # Initialize Population
    pop = Population(p["max_iter"], p["satisfactory"], p["stagnation_limit"], p["stagnation_diff"],
                     p["pop_size"], p["win_prob"], p["init_sigma"], p["tournament_participants"], p["carry_over"])
    # return RNASeqDeconvolution initialized with pop
    return RNASeqDeconvolution(pop)

def compare(var1:str, start1:float, step1:float, end1:float,
            var2:str|None = None, start2:float|None = None, step2:float|None = None, end2:float|None = None) \
            -> list[pd.DataFrame]:
    """
    Generate a comparison matrix for different values of one or two variables.
    :param var1: name of variable to change
    :param start1: start value of var1
    :param step1: step of var1 value
    :param end1: end value of var1 (inclusive)
    :param var2: name of variable to change (if None, will show a matrix only for var1)
    :param start2: start value of var2
    :param step2: step of var2 value
    :param end2: end value of var2 (inclusive)
    :return: list of Pandas dataframes.
             If var2 isn't given, produce a single matrix with rows representing value
             and columns representing solution fitness score, number of iterations it took to get there and stop reason.
             If var2 IS given, produce 3 matrices for fitness score, iterations and stop reason where
             rows represent var1's value and columns represent var2's value.
    """
    # initialize var1 values and labels
    var1_values = list([start1])
    var1_labels = list([var1 + " = " + str(start1)])
    while var1_values[-1] < end1:
        # var1_values += var1_values[-1] + step1
        # var1_labels += var1 + " = " + str(var1_values[-1])
        var1_values.append(var1_values[-1] + step1)
        var1_labels.append(var1 + " = " + str(var1_values[-1]))
    # if var2 isn't specified
    if var2 is None: # only testing var1, everything else is as default
        result_matrix = []
        pbar = tqdm(range(len(var1_values)), total=len(var1_values), dynamic_ncols=True, position=0)
        for i in pbar: # for each value
            pbar.set_postfix_str(var1_labels[i])
            experiment = set_parameters({var1: var1_values[i]}) # set experiment to use current var1 value
            experiment.run(False, 1) # run experiment
            result_matrix += [[experiment.result_fitness_score, experiment.pop.current_iter,
                               experiment.detect_stop_reason()]] #extract fitness score, iteration and stop reason
        output = pd.DataFrame(result_matrix, var1_labels, RESULT_LABELS) # wrap in pandas DataFrame
        return [output] # wrap output in list of size 1

    # if var2 IS specified
    # initialize var2 values and labels
    fitness_matrix, iteration_matrix, stop_reason_matrix = [], [], []
    var2_values = list([start2])
    var2_labels = list([var2 + " = " + str(start2)])
    while var2_values[-1] < end2:
        # var2_values += var2_values[-1] + step2
        # var2_labels += var2 + " = " + str(var2_values[-1])
        var2_values.append(var2_values[-1] + step2)
        var2_labels.append(var2 + " = " + str(var2_values[-1]))
    # testing all (var1 x var2) pairs
    pbar1 = trange(len(var1_values), total=len(var1_values), dynamic_ncols=True, position=0)
    pbar2 = trange(len(var2_values), total=len(var2_values), dynamic_ncols=True, position=1)
    for i in pbar1: # for each var1 value
        pbar1.set_postfix_str(var1_labels[i])
        # initialize result rows
        fitness_row, iteration_row, stop_reason_row = [], [], []
        for j in pbar2: # for each var2 value
            pbar2.set_postfix_str(var2_labels[j])
            # run experiment
            experiment = set_parameters({var1: var1_values[i], var2: var2_values[j]})
            experiment.run(False, 2)
            # update result rows
            fitness_row += experiment.result_fitness_score
            iteration_row += experiment.pop.current_iter
            stop_reason_row += experiment.detect_stop_reason()
        # finished checking all var2 values for this specific var1 value, update matrices with the rows we got
        fitness_matrix += [fitness_row]
        iteration_matrix += [iteration_row]
        stop_reason_matrix += [stop_reason_row]

    # wrap in DataFrame and output
    fitness_df = pd.DataFrame(fitness_matrix, var1_labels, var2_labels)
    iteration_df = pd.DataFrame(iteration_matrix, var1_labels, var2_labels)
    stop_reason_df = pd.DataFrame(stop_reason_matrix, var1_labels, var2_labels)
    return list([fitness_df, iteration_df, stop_reason_df])


if __name__ == "__main__":
    M = parse_input_matrix(DEFAULT_PARAMS.get("M_path"))
    H = parse_input_matrix(DEFAULT_PARAMS.get("H_path"))
    DEFAULT_PARAMS["M"], DEFAULT_PARAMS["H"] = M, H
    # params = {"M": M, "H": H}
    # experiment = set_parameters(params)
    # experiment.run()
    # # TODO: what comes next is just a quick experiment to see it works
    # print(experiment.result)
    # print(experiment.result_fitness_score)
    df_arr = compare("mut_prob", 0.005, 0.005, 0.1)
    for df in df_arr:
        print(df)

