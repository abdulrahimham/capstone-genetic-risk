"""
gene_prs.py: week 4.

1. cis-eQTL scan for every gene on one chromosome (week 3 scaled up).
2. A polygenic score for a gene's expression, built from its cis-eQTL
   results by pruning and thresholding (P+T): clump the cis SNPs once, then
   score people using only the clumped SNPs below each p-value threshold.
3. Cross-validation so the score's accuracy isn't measured on the same
   people it was fit on:
     - training (80%): run the cis-eQTL regression -> effect sizes and p values
     - tuning (10%):   try each p-value threshold, keep the one with the best R^2
     - testing (10%):  report R^2 of the chosen score in people never used for fitting
   R^2 here is the squared Pearson correlation between predicted and true expression.
"""

import numpy as np
import pandas as pd

from eqtl import cis_window, regress_each_snp
from prs import clump_chromosome


def split_samples(sample_ids, train_fraction, tune_fraction, seed):
    """Random train / tune / test split of individual IDs."""
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sample_ids))
    n_train = int(round(train_fraction * len(shuffled)))
    n_tune = int(round(tune_fraction * len(shuffled)))
    return shuffled[:n_train], shuffled[n_train:n_train + n_tune], shuffled[n_train + n_tune:]


def gene_windows(annotation, chromosome, bim, window_bp):
    """For each gene on the chromosome, the column indices of its cis SNPs in the .bim file."""
    positions = bim["position_bp"].to_numpy()
    order = np.argsort(positions)
    sorted_positions = positions[order]
    genes = annotation[annotation["Chr"] == str(chromosome)]
    windows = {}
    for target_id, coordinate in genes["Coord"].items():
        start, end = cis_window(int(coordinate), window_bp)
        left = np.searchsorted(sorted_positions, start, side="left")
        right = np.searchsorted(sorted_positions, end, side="right")
        windows[target_id] = np.sort(order[left:right])
    return windows


def correlation_squared(predicted, observed):
    """Squared Pearson correlation; 0 when the prediction is constant (no SNPs passed)."""
    if np.std(predicted) == 0 or np.std(observed) == 0:
        return 0.0, np.nan
    r = float(np.corrcoef(predicted, observed)[0, 1])
    return r ** 2, r


def chromosome_cis_eqtl(dosages, positions, snp_ids, expression_matrix, windows, gene_symbols=None):
    """
    cis-eQTL scan for every gene on a chromosome.

    dosages: individuals x all SNPs on the chromosome.
    expression_matrix: genes x individuals (same individual order as dosages).
    windows: target_id -> SNP column indices.
    Returns (all_results, per_gene): every gene-SNP test, and one summary row per gene.
    """
    all_parts, gene_rows = [], []
    for target_id, snp_columns in windows.items():
        if len(snp_columns) == 0:
            continue
        expression_values = expression_matrix.loc[target_id].to_numpy(dtype=float)
        stats_table = regress_each_snp(expression_values, dosages[:, snp_columns])
        stats_table.insert(0, "target_id", target_id)
        stats_table.insert(1, "snp_id", snp_ids[snp_columns])
        stats_table.insert(2, "position_bp", positions[snp_columns])
        all_parts.append(stats_table)

        n_tested = int(stats_table["p_value"].notna().sum())
        best = stats_table.loc[stats_table["p_value"].idxmin()] if n_tested else None
        bonferroni = 0.05 / n_tested if n_tested else np.nan
        gene_rows.append({
            "target_id": target_id,
            "gene_symbol": gene_symbols.get(target_id, target_id) if gene_symbols else target_id,
            "n_cis_snps": len(snp_columns),
            "n_tested_snps": n_tested,
            "lead_snp": best["snp_id"] if best is not None else None,
            "lead_p_value": float(best["p_value"]) if best is not None else np.nan,
            "lead_r_squared": float(best["r_squared"]) if best is not None else np.nan,
            "n_bonferroni_significant": int((stats_table["p_value"] < bonferroni).sum()),
            "is_egene_bonferroni": bool((stats_table["p_value"] < bonferroni).any()),
        })
    return pd.concat(all_parts, ignore_index=True), pd.DataFrame(gene_rows)


def fit_gene_score(train_dosages, train_expression, positions, clump_kb, clump_r2):
    """
    Effect sizes from training data, then clumping (LD measured in the training
    people). Returns a table of clumped SNPs with beta and p value.
    """
    stats_table = regress_each_snp(train_expression, train_dosages)
    stats_table["position_bp"] = positions
    stats_table["column"] = np.arange(len(positions))
    testable = stats_table.dropna(subset=["p_value"]).reset_index(drop=True)
    if testable.empty:
        return testable
    testable = testable.rename(columns={"p_value": "P"})
    kept = clump_chromosome(testable, train_dosages[:, testable["column"].to_numpy()], clump_kb, clump_r2)
    return testable[kept].reset_index(drop=True)


def predict(dosages, clumped, p_threshold):
    """Score = sum of beta x dosage over clumped SNPs with p below the threshold."""
    chosen = clumped[clumped["P"] < p_threshold] if p_threshold < 1 else clumped
    if chosen.empty:
        return np.zeros(dosages.shape[0]), 0
    return dosages[:, chosen["column"].to_numpy()] @ chosen["beta"].to_numpy(), len(chosen)


def threshold_table(dosages, expression_values, clumped, p_thresholds):
    """R^2 at each threshold for one group of people."""
    rows = []
    for threshold in p_thresholds:
        predicted, n_snps = predict(dosages, clumped, threshold)
        r_squared, r = correlation_squared(predicted, expression_values)
        rows.append({"p_threshold": threshold, "n_snps": n_snps, "r": r, "r_squared": r_squared})
    return pd.DataFrame(rows)


def cross_validate_gene(dosages_by_group, expression_by_group, positions, p_thresholds, clump_kb, clump_r2):
    """
    Train on 'train', pick the threshold on 'tune', report accuracy on 'test'.
    dosages_by_group / expression_by_group: dicts keyed by 'train', 'tune', 'test'.
    Returns (summary dict, tuning table).
    """
    clumped = fit_gene_score(dosages_by_group["train"], expression_by_group["train"], positions, clump_kb, clump_r2)
    tuning = threshold_table(dosages_by_group["tune"], expression_by_group["tune"], clumped, p_thresholds)
    # Ties (e.g. several thresholds keeping the same SNPs) go to the stricter threshold.
    best_row = tuning.sort_values(["r_squared", "p_threshold"], ascending=[False, True]).iloc[0]
    test_predicted, n_snps = predict(dosages_by_group["test"], clumped, best_row["p_threshold"])
    test_r_squared, test_r = correlation_squared(test_predicted, expression_by_group["test"])
    summary = {
        "n_clumped_snps": len(clumped),
        "best_p_threshold": float(best_row["p_threshold"]),
        "tune_r_squared": float(best_row["r_squared"]),
        "n_snps_in_score": n_snps,
        "test_r": test_r,
        "test_r_squared": test_r_squared,
    }
    return summary, tuning
