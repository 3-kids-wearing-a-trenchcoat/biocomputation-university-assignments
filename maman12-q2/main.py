from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd
from pathlib import Path
from Individual import Individual, FTYPE
from Population import Population
from RNASeqDeconvolution import RNASeqDeconvolution

# default values
DEFAULT_PARAMS = {"rng": np.random.default_rng(123),
                  "mut_prob": 0.1,
                  "mut_standard_deviation": 0.2,
                  "crossover_prob": 0.8,
                  "max_iter": 10000,
                  "satisfactory": 1e-3,
                  "stagnation_limit": 200,
                  "stagnation_diff": 1e-3,
                  "pop_size": 1000,
                  "win_prob": 0.7,
                  "init_sigma": 0.7,
                  "tournament_participants": 3,
                  "carry_over": 2,
                  "H_path": Path(__file__).parent / "matrices" / "gene_celltype_TPM.tsv",
                  "M_path": Path(__file__).parent / "matrices" / "gene_sample_TPM.tsv",
                  "true_result_path": Path(__file__).parent / "matrices" / "sample_celltype_groundT.tsv"
                  }

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


if __name__ == "__main__":
    M = parse_input_matrix(DEFAULT_PARAMS.get("M_path"))
    H = parse_input_matrix(DEFAULT_PARAMS.get("H_path"))
    params = {"M": M, "H": H}
    experiment = set_parameters(params)
    experiment.run()
    # TODO: what comes next is just a quick experiment to see it works
    print(experiment.result)
    print(experiment.result_fitness_score)
