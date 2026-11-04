"""
eqtl.py: cis-eQTL mapping for a single gene (week 3).

For one gene, take every SNP within `window_bp` of the gene's coordinate and
fit a simple linear regression of expression on genotype dosage,
expression ~ genotype, one SNP at a time. This matches the course's
suggested lm(y_gene_expression ~ genotype_snp_j).
"""

import numpy as np
import pandas as pd
from scipy import stats

from etl import read_bim, read_fam, read_genotypes, shared_sample_ids


def lookup_gene(annotation, gene_id):
    """Find a gene by Ensembl ID, with or without the version suffix (e.g. ENSG00000164308)."""
    if gene_id in annotation.index:
        return gene_id, annotation.loc[gene_id]
    matches = [target for target in annotation.index if target.split(".")[0] == gene_id]
    if not matches:
        raise KeyError(f"Gene {gene_id} not found in the expression data")
    return matches[0], annotation.loc[matches[0]]


def cis_window(gene_coordinate, window_bp):
    """Start and end (bp) of the cis window around a gene coordinate."""
    return max(0, gene_coordinate - window_bp), gene_coordinate + window_bp


def regress_each_snp(expression_values, dosages):
    """
    Simple linear regression of expression on each SNP's dosage.
    Returns slope (beta), standard error, t statistic, p value and R^2 per SNP.
    SNPs with no variation in this sample get NaN.
    """
    n_individuals = len(expression_values)
    expression_centered = expression_values - expression_values.mean()
    dosages_centered = dosages - dosages.mean(axis=0)

    sum_sq_genotype = (dosages_centered ** 2).sum(axis=0)
    sum_sq_expression = (expression_centered ** 2).sum()
    cross_product = dosages_centered.T @ expression_centered

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = cross_product / sum_sq_genotype
        r_squared = cross_product ** 2 / (sum_sq_genotype * sum_sq_expression)
        residual_variance = sum_sq_expression * (1 - r_squared) / (n_individuals - 2)
        standard_error = np.sqrt(residual_variance / sum_sq_genotype)
        t_statistic = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_statistic), df=n_individuals - 2)
    return pd.DataFrame({
        "beta": beta,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "r_squared": r_squared,
    })


def cis_eqtl_single_gene(genotype_dir, file_prefix, annotation, expression, gene_id, window_bp):
    """
    Run cis-eQTL mapping for one gene.

    Returns (results, gene_info): results has one row per cis SNP with
    position and regression statistics, sorted by position; gene_info is a
    dict describing the gene, window and sample size.
    """
    target_id, gene_row = lookup_gene(annotation, gene_id)
    chromosome = gene_row["Chr"]
    gene_coordinate = int(gene_row["Coord"])
    window_start, window_end = cis_window(gene_coordinate, window_bp)

    bim = read_bim(genotype_dir, file_prefix, chromosome)
    in_window = bim["position_bp"].between(window_start, window_end).to_numpy()

    fam = read_fam(genotype_dir, file_prefix, chromosome)
    sample_ids = shared_sample_ids(fam, expression)
    dosages = read_genotypes(genotype_dir, file_prefix, chromosome, snp_mask=in_window, sample_ids=sample_ids)
    expression_values = expression.loc[target_id, sample_ids].to_numpy(dtype=float)

    statistics = regress_each_snp(expression_values, dosages)
    snp_info = bim.loc[in_window, ["snp_id", "position_bp", "allele_1", "allele_2"]].reset_index(drop=True)
    results = pd.concat([snp_info, statistics], axis=1)
    results["neg_log10_p"] = -np.log10(results["p_value"])

    gene_info = {
        "target_id": target_id,
        "chromosome": str(chromosome),
        "gene_coordinate": gene_coordinate,
        "window_start": window_start,
        "window_end": window_end,
        "n_individuals": len(sample_ids),
        "n_cis_snps": int(in_window.sum()),
        "n_tested_snps": int(results["p_value"].notna().sum()),
    }
    return results, gene_info


def summarize_top_hits(results, gene_info, n_top=10):
    """Bonferroni threshold across the gene's cis SNPs and the top associated SNPs."""
    n_tests = gene_info["n_tested_snps"]
    bonferroni_threshold = 0.05 / n_tests if n_tests else np.nan
    top = results.nsmallest(n_top, "p_value")
    n_significant = int((results["p_value"] < bonferroni_threshold).sum())
    return top, bonferroni_threshold, n_significant


def snps_tied_with_lead(results):
    """
    SNPs whose p value equals the lead SNP's. These have identical genotypes in
    this sample (perfect linkage), so the data cannot say which one is causal.
    """
    lead_p_value = results["p_value"].min()
    return results.loc[np.isclose(results["p_value"], lead_p_value, rtol=1e-9, atol=0), "snp_id"].tolist()
