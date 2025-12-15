from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, DTypeLike
import pandas as pd
from pathlib import Path
from Individual import Individual, FTYPE
from Population import Population
from RNASeqDeconvolution import RNASeqDeconvolution
from Niches import Niches
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr

# default values
DEFAULT_PARAMS = {"rng_seed": 123,
                  "mut_prob": 0.05,
                  "mut_standard_deviation": 1.5,
                  "crossover_prob": 0.8,
                  "max_iter": 2000,
                  "satisfactory": 1e-5,
                  "stagnation_limit": 200,
                  "stagnation_diff": 1e-6,
                  "pop_size": 200,
                  "win_prob": 0.9,
                  "init_sigma": 2.5,
                  "tournament_participants": 2,
                  "carry_over": 10,
                  "H_path": Path(__file__).parent / "matrices" / "gene_celltype_TPM.tsv",
                  "M_path": Path(__file__).parent / "matrices" / "gene_sample_TPM.tsv",
                  "true_result_path": Path(__file__).parent / "matrices" / "sample_celltype_groundT_sorted_redacted_with_unclassified.tsv",
                  "niche_num": 2,
                  "migration_interval": 200,
                  "migrator_num": 5,
                  "use_marker_genes": True,
                  "top_n_marker_genes": 200,
                  "min_mean_marker_genes": 1e-6,
                  "log_transform_marker_genes": True
                  }
RESULT_LABELS = ["fitness score", "iterations", "cause of stop"]

def parse_input_matrix(path: Path, t: DTypeLike = FTYPE) -> NDArray:
    """
    turn the matrix in the given path into a numpy matrix, discarding row/column names
    :param path: path to file (using pathlib)
    :param t: dtype of output matrix (defaults to FTYPE as defined in Individual)
    :return: NDArray of type t
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    output = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=t)
    return output

# adding values to DEFAULT_PARAMS here because I need parse_input_matrix
M = parse_input_matrix(DEFAULT_PARAMS.get("M_path"))
H = parse_input_matrix(DEFAULT_PARAMS.get("H_path"))
DEFAULT_PARAMS["M"], DEFAULT_PARAMS["H"] = M, H
TRUE = parse_input_matrix(DEFAULT_PARAMS.get("true_result_path")).transpose()
DEFAULT_PARAMS["TRUE"] = TRUE / 100

def set_parameters(input_params: dict[str, float | int | NDArray[FTYPE] | np.random.Generator], calc_mean=False) -> RNASeqDeconvolution:
    """
    Helper function which initializes the **many** variables across Population and Individual.
    Each variable-value pair in params has that value set for that variable.
    Every variable not in params is set to the default
    :param input_params: dict[str, Any] - key is variable name as a string, value is the value to assign to the variable
    :param calc_mean: whether to calculate the mean value at each iteration, reduces performance
    :return: RNASeqDeconvolution object with the variables set as specified
    """
    # start with parameters as specified in defaults
    p = DEFAULT_PARAMS.copy()
    # change/add all specified key-value pairs in input_params
    for key, value in input_params.items():
        p[key] = value

    p["rng"] = np.random.default_rng(p["rng_seed"])

    # set static Individual parameters
    Individual.set_static_vars(p["rng"], p["mut_prob"], p["mut_standard_deviation"],
                               p["crossover_prob"], p["M"], p["H"], p["TRUE"],
                               p["use_marker_genes"], p["top_n_marker_genes"],
                               p["min_mean_marker_genes"], p["log_transform_marker_genes"])
    # Initialize Population
    pop = None
    if p["niche_num"] < 2: # if less than 2 niches, it's a regular, undivided population
        pop = Population(p["max_iter"], p["satisfactory"], p["stagnation_limit"], p["stagnation_diff"],
                         p["pop_size"], p["win_prob"], p["init_sigma"], p["tournament_participants"], p["carry_over"],
                         calc_mean)
    else:
        pop = Niches(p["max_iter"], p["satisfactory"], p["stagnation_limit"], p["stagnation_diff"],
                     p["pop_size"], p["win_prob"], p["init_sigma"], p["tournament_participants"], p["carry_over"],
                     calc_mean, p["niche_num"], p["migration_interval"], p["migrator_num"])
    # return RNASeqDeconvolution initialized with pop
    return RNASeqDeconvolution(pop)

def get_true_results() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_PARAMS["true_result_path"], sep="\t", index_col=0, header=0)
    return df

def compare_to_true_results(phen:NDArray[FTYPE]) -> pd.DataFrame:
    """
    Produce a DataFrame matrix of difference between result we got and the true result
    :param phen: numpy matrix of result phenotype
    :return: pandas DataFrame of difference when subtracting true result from result we got.
             labeled in the same way as the true results
    """
    result = phen.transpose() * 100 # convert form fractional ([0,1]) to percentage ([0,100]) representation
    true_result_df = get_true_results()
    # calculate difference between phenotype we got and true phenotype
    true_result_phenotype = true_result_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=FTYPE)
    diff = result - true_result_phenotype
    # extract labels from true_result_df
    col_labels = true_result_df.columns.to_list()
    row_labels = true_result_df.index.to_list()
    # produce new DataFrame object for diff
    df_diff = pd.DataFrame(diff, columns=col_labels, index=row_labels)
    return df_diff


def evaluate_deconvolution_solution(X_hat, X_true, H, M, L=None, name="candidate", top_genes=10, eps=1e-12):
    """
    Evaluate a candidate deconvolution solution (X_hat) vs ground truth and data.
    - Automatically displays a readable pandas DataFrame report in Jupyter.
    - Returns a metrics dict that includes the DataFrames under keys:
        'report_df', 'per_cell_df', 'per_sample_df'.

    Signature preserved.
    """
    # ---- input sanity checks / cast ----
    X_hat = np.asarray(X_hat, dtype=float)
    X_true = np.asarray(X_true, dtype=float)
    H = np.asarray(H, dtype=float)
    M = np.asarray(M, dtype=float)

    G, K = H.shape
    G2, S = M.shape
    K2, S2 = X_hat.shape
    assert G2 == G, f"H and M must share gene dimension (G). got {G} vs {G2}"
    assert X_true.shape == (K, S), f"X_true must be shape (K,S). got {X_true.shape}"
    assert K2 == K, f"X_hat has K={K2} but H has K={K}"

    if L is None:
        L = M.sum(axis=0)
    L = np.asarray(L).reshape(-1)
    assert L.shape[0] == S, "L length must equal number of samples S"

    # ---- predictions ----
    mu = (H @ X_hat) * L.reshape(1, -1)               # G x S
    mu_true = (H @ X_true) * L.reshape(1, -1)

    # ---- residuals and basic metrics ----
    resid = M - mu
    resid_true = M - mu_true
    fro_candidate = float(np.sum(resid**2))
    fro_true = float(np.sum(resid_true**2))
    RMSE_gene = float(np.sqrt(fro_candidate / (G * S)))
    RMSE_gene_true = float(np.sqrt(fro_true / (G * S)))
    max_abs_resid = float(np.max(np.abs(resid)))
    mean_abs_resid = float(np.mean(np.abs(resid)))

    # ---- Jensen-Shannon per sample (on relative columns) ----
    P = M / (L.reshape(1, -1) + eps)
    Q = mu / (L.reshape(1, -1) + eps)
    js_per_sample = []
    for j in range(S):
        p = P[:, j]
        q = Q[:, j]
        p_norm = p / (p.sum() + eps)
        q_norm = q / (q.sum() + eps)
        js_dist = distance.jensenshannon(p_norm, q_norm)
        js_per_sample.append(float(js_dist**2))
    mean_js = float(np.mean(js_per_sample))
    median_js = float(np.median(js_per_sample))

    # ---- X composition metrics ----
    diff_X = X_hat - X_true
    per_entry_rmse = float(np.sqrt(np.mean(diff_X**2)))
    per_entry_mae = float(np.mean(np.abs(diff_X)))
    mae_per_cell = np.mean(np.abs(diff_X), axis=1)   # length K
    rmse_per_cell = np.sqrt(np.mean(diff_X**2, axis=1))
    mae_overall = float(np.mean(mae_per_cell))
    rmse_overall = float(np.mean(rmse_per_cell))

    # correlations per cell-type (safe)
    pearson_by_cell = []
    spearman_by_cell = []
    for k in range(K):
        a = X_hat[k, :]
        b = X_true[k, :]
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            pearson_by_cell.append(np.nan)
            spearman_by_cell.append(np.nan)
        else:
            try:
                pearson_by_cell.append(float(pearsonr(a, b)[0]))
            except Exception:
                pearson_by_cell.append(np.nan)
            try:
                spearman_by_cell.append(float(spearmanr(a, b)[0]))
            except Exception:
                spearman_by_cell.append(np.nan)
    mean_pearson = float(np.nanmean(pearson_by_cell))
    mean_spearman = float(np.nanmean(spearman_by_cell))

    mae_per_sample = np.mean(np.abs(diff_X), axis=0)
    rmse_per_sample = np.sqrt(np.mean(diff_X**2, axis=0))

    # ---- gene-level explained variance ----
    var_M_by_gene = np.var(M, axis=1, ddof=0)
    var_resid_by_gene = np.var(resid, axis=1, ddof=0)
    explained_var_by_gene = 1.0 - np.divide(var_resid_by_gene, (var_M_by_gene + eps))
    mean_explained_var = float(np.nanmean(explained_var_by_gene))

    # ---- top residual genes ----
    mean_abs_resid_per_gene = np.mean(np.abs(resid), axis=1)
    top_genes_idx_overall = np.argsort(mean_abs_resid_per_gene)[-top_genes:][::-1]
    top_genes_per_sample = []
    for j in range(S):
        idxs = np.argsort(np.abs(resid[:, j]))[-top_genes:][::-1]
        top_genes_per_sample.append(idxs)

    # ---- conditioning info for H ----
    try:
        H_colnorm = H / (H.sum(axis=0, keepdims=True) + eps)
        svals = np.linalg.svd(H_colnorm, compute_uv=False)
        tol = svals.max() * max(H_colnorm.shape) * np.finfo(float).eps
        numeric_rank = int(np.sum(svals > tol))
        cond_est = float(svals[0] / (svals[-1] + eps)) if svals.size > 1 else float(np.inf)
    except Exception:
        numeric_rank = None
        cond_est = None

    # ---- Build pandas DataFrames for return & display ----
    summary = {
        "name": name,
        "G": G, "K": K, "S": S,
        "fro_candidate": fro_candidate,
        "fro_true": fro_true,
        "rmse_gene_candidate": RMSE_gene,
        "rmse_gene_true": RMSE_gene_true,
        "mean_abs_resid": mean_abs_resid,
        "max_abs_resid": max_abs_resid,
        "mean_js": mean_js,
        "per_entry_rmse_X": per_entry_rmse,
        "per_entry_mae_X": per_entry_mae,
        "mae_overall": mae_overall,
        "rmse_overall": rmse_overall,
        "mean_pearson": mean_pearson,
        "mean_spearman": mean_spearman,
        "H_numeric_rank": numeric_rank,
        "H_cond_est": cond_est,
        "mean_explained_var": mean_explained_var
    }
    summary_df = pd.DataFrame([summary])

    per_cell_df = pd.DataFrame({
        "cell_index": np.arange(K),
        "mae_per_cell": mae_per_cell,
        "rmse_per_cell": rmse_per_cell,
        "pearson_by_cell": pearson_by_cell,
        "spearman_by_cell": spearman_by_cell
    })

    per_sample_df = pd.DataFrame({
        "sample_index": np.arange(S),
        "mae_per_sample": mae_per_sample,
        "rmse_per_sample": rmse_per_sample,
        "js_per_sample": js_per_sample
    })

    # Display nicely in Jupyter if possible
    try:
        from IPython.display import display
        # format floats compactly for display only
        with pd.option_context("display.float_format", "{:0.6g}".format):
            print("\n=== Evaluation summary ===")
            display(summary_df.T)   # vertical layout
            print("\n=== Per-cell metrics (first 12 rows) ===")
            display(per_cell_df.head(12))
            print("\n=== Per-sample metrics ===")
            display(per_sample_df)
            print("\n=== Top residual genes (overall) ===")
            display(pd.Series(top_genes_idx_overall, name="top_residual_gene_idx"))
    except Exception:
        # fallback: print textual summaries
        print("\n=== Evaluation summary ===")
        print(summary_df.T.to_string())
        print("\n=== Per-cell metrics (first 12 rows) ===")
        print(per_cell_df.head(12).to_string(index=False))
        print("\n=== Per-sample metrics ===")
        print(per_sample_df.to_string(index=False))
        print("\n=== Top residual genes (overall) ===")
        print(top_genes_idx_overall.tolist())

    # ---- prepare return dict including DataFrames ----
    metrics = {
        "fro_candidate": fro_candidate,
        "fro_true": fro_true,
        "rmse_gene_candidate": RMSE_gene,
        "rmse_gene_true": RMSE_gene_true,
        "mean_abs_resid": mean_abs_resid,
        "max_abs_resid": max_abs_resid,
        "mean_js": mean_js,
        "per_entry_rmse_X": per_entry_rmse,
        "per_entry_mae_X": per_entry_mae,
        "mae_per_cell": mae_per_cell,
        "rmse_per_cell": rmse_per_cell,
        "mae_overall": mae_overall,
        "rmse_overall": rmse_overall,
        "pearson_by_cell": pearson_by_cell,
        "spearman_by_cell": spearman_by_cell,
        "mean_pearson": mean_pearson,
        "mean_spearman": mean_spearman,
        "mae_per_sample": mae_per_sample,
        "rmse_per_sample": rmse_per_sample,
        "explained_var_by_gene_mean": mean_explained_var,
        "top_genes_overall": top_genes_idx_overall,
        "top_genes_per_sample": top_genes_per_sample,
        "H_numeric_rank": numeric_rank,
        "H_cond_est": cond_est,
        # include DataFrames for convenience
        "report_df": summary_df,
        "per_cell_df": per_cell_df,
        "per_sample_df": per_sample_df
    }

    return metrics