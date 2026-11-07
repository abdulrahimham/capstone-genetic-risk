"""
prs.py: polygenic risk score (PRS) by clumping and thresholding (C+T), week 5.

Steps, following the PRS tutorial (choishingwan.github.io/PRS-Tutorial):
1. QC the GWAS summary statistics: keep INFO > 0.8 and MAF > 0.01, drop
   strand-ambiguous SNPs (A/T, C/G) and duplicated SNP IDs, convert OR to beta.
2. Match the GWAS SNPs to the genotype panel and line up the effect allele.
3. Clump: walk SNPs from smallest to largest p value; each kept SNP removes
   every other SNP within `clump_kb` with LD r^2 > `clump_r2` (LD measured in
   the 1000 Genomes panel).
4. Threshold: keep clumped SNPs with p < `p_threshold`.
5. Score each person: sum over kept SNPs of beta x (copies of the effect allele).

Only SNPs with p < p_threshold can end up in the final score, and whether a SNP
survives clumping depends only on SNPs with smaller p values. So clumping just
those SNPs gives exactly the same result as clumping everything first.
"""

import numpy as np
import pandas as pd

from etl import read_bim, read_genotypes

AMBIGUOUS_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def qc_summary_statistics(gwas, min_info, min_maf):
    """Basic summary-statistic QC from the PRS tutorial; adds a BETA column = log(OR)."""
    n_start = len(gwas)
    gwas = gwas[(gwas["INFO"] > min_info) & (gwas["MAF"] > min_maf)].copy()
    gwas["A1"] = gwas["A1"].str.upper()
    gwas["A2"] = gwas["A2"].str.upper()
    is_ambiguous = [(a1, a2) in AMBIGUOUS_PAIRS for a1, a2 in zip(gwas["A1"], gwas["A2"])]
    gwas = gwas[~np.array(is_ambiguous)]
    gwas = gwas[~gwas["SNP"].duplicated(keep=False)]
    gwas["BETA"] = np.log(gwas["OR"])
    qc_counts = {"gwas_snps_input": n_start, "gwas_snps_after_qc": len(gwas)}
    return gwas.reset_index(drop=True), qc_counts


def align_to_panel(gwas, bim):
    """
    Join GWAS SNPs to the genotype panel by rsID and express each effect as
    "per copy of the panel's allele_1" (the allele the dosages count).

    Handles alleles listed in the other order or on the other strand; drops
    SNPs whose alleles don't match at all. Returns the joined table with a
    `beta_panel` column.
    """
    merged = gwas.merge(bim, left_on="SNP", right_on="snp_id", how="inner")
    gwas_a1, gwas_a2 = merged["A1"], merged["A2"]
    flipped_a1 = gwas_a1.str.translate(COMPLEMENT)
    flipped_a2 = gwas_a2.str.translate(COMPLEMENT)
    panel_a1, panel_a2 = merged["allele_1"], merged["allele_2"]

    same = ((gwas_a1 == panel_a1) & (gwas_a2 == panel_a2)) | ((flipped_a1 == panel_a1) & (flipped_a2 == panel_a2))
    swapped = ((gwas_a1 == panel_a2) & (gwas_a2 == panel_a1)) | ((flipped_a1 == panel_a2) & (flipped_a2 == panel_a1))

    merged = merged[same | swapped].copy()
    merged["beta_panel"] = np.where(same[same | swapped], merged["BETA"], -merged["BETA"])
    return merged.reset_index(drop=True)


def clump_chromosome(snps, dosages, clump_kb, clump_r2):
    """
    Greedy LD clumping on one chromosome.

    snps: table with `position_bp` and `P`, one row per column of `dosages`.
    dosages: individuals x SNPs matrix from the LD reference panel.
    Returns a boolean array marking the index SNPs that are kept.
    """
    standardized = dosages - dosages.mean(axis=0)
    norms = np.linalg.norm(standardized, axis=0)
    norms[norms == 0] = np.inf
    standardized = standardized / norms

    positions = snps["position_bp"].to_numpy()
    order = np.argsort(snps["P"].to_numpy(), kind="stable")
    window_bp = clump_kb * 1000
    available = np.ones(len(snps), dtype=bool)
    kept = np.zeros(len(snps), dtype=bool)

    for index_snp in order:
        if not available[index_snp]:
            continue
        kept[index_snp] = True
        available[index_snp] = False
        nearby = np.flatnonzero(available & (np.abs(positions - positions[index_snp]) <= window_bp))
        if len(nearby):
            r_squared = (standardized[:, nearby].T @ standardized[:, index_snp]) ** 2
            available[nearby[r_squared > clump_r2]] = False
    return kept


def select_prs_snps(gwas, genotype_dir, file_prefix, chromosomes, p_threshold, clump_kb, clump_r2, keep_snps=None):
    """
    Run alignment, thresholding and clumping across chromosomes.

    keep_snps: optional set of rsIDs to restrict to before clumping (for
    example, SNPs that the 23andMe chip also measured), so every person is
    scored on the same SNPs.
    Returns (selected, counts): one row per SNP in the final score, and a
    dict of how many SNPs survived each step.
    """
    selected_parts = []
    counts = {"aligned_to_panel": 0, "below_threshold": 0, "after_clumping": 0}
    for chromosome in chromosomes:
        bim = read_bim(genotype_dir, file_prefix, chromosome)
        bim["panel_index"] = np.arange(len(bim))
        chrom_gwas = gwas[gwas["CHR"].astype(str) == str(chromosome)]
        aligned = align_to_panel(chrom_gwas, bim)
        if keep_snps is not None:
            aligned = aligned[aligned["SNP"].isin(keep_snps)]
        counts["aligned_to_panel"] += len(aligned)

        candidates = aligned[aligned["P"] < p_threshold].sort_values("panel_index").reset_index(drop=True)
        counts["below_threshold"] += len(candidates)
        if candidates.empty:
            continue

        snp_mask = np.zeros(len(bim), dtype=bool)
        snp_mask[candidates["panel_index"].to_numpy()] = True
        dosages = read_genotypes(genotype_dir, file_prefix, chromosome, snp_mask=snp_mask)
        kept = clump_chromosome(candidates, dosages, clump_kb, clump_r2)
        selected_parts.append(candidates[kept])
        counts["after_clumping"] += int(kept.sum())

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected[["CHR", "SNP", "position_bp", "allele_1", "allele_2", "beta_panel", "P", "panel_index"]], counts


def score_panel(selected, genotype_dir, file_prefix):
    """PRS for every individual in the genotype panel: sum of beta x dosage over the selected SNPs."""
    total_score = None
    sample_ids = None
    for chromosome, chrom_snps in selected.groupby("CHR"):
        bim_length = len(read_bim(genotype_dir, file_prefix, chromosome))
        snp_mask = np.zeros(bim_length, dtype=bool)
        snp_mask[chrom_snps["panel_index"].to_numpy()] = True
        dosages = read_genotypes(genotype_dir, file_prefix, chromosome, snp_mask=snp_mask)
        chrom_score = dosages @ chrom_snps.sort_values("panel_index")["beta_panel"].to_numpy()
        total_score = chrom_score if total_score is None else total_score + chrom_score
        if sample_ids is None:
            from etl import read_fam
            sample_ids = read_fam(genotype_dir, file_prefix, chromosome)["individual_id"].tolist()
    return pd.DataFrame({"individual_id": sample_ids, "prs": total_score})


def place_in_distribution(person_score, panel_scores):
    """Percentile and z-score of one person's PRS relative to the panel."""
    percentile = 100 * (panel_scores < person_score).mean()
    z_score = (person_score - panel_scores.mean()) / panel_scores.std(ddof=1)
    return {"prs": float(person_score), "percentile": float(percentile), "z_score": float(z_score)}
