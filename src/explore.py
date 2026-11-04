"""
explore.py: descriptive checks on the raw data (weeks 1 and 2).

- Describes the genotype and expression files (week 1).
- Bins variants by minor allele frequency (MAF) into rare, low frequency
  and common, using the course's cutoffs (week 2).
- Counts the individuals present in both datasets (week 2).
"""

import os

import numpy as np
import pandas as pd

from etl import read_bim, read_fam, read_expression, read_genotypes, shared_sample_ids


def minor_allele_frequency(dosages):
    """MAF per SNP from an individuals x SNPs dosage matrix."""
    allele_1_frequency = dosages.mean(axis=0) / 2
    return np.minimum(allele_1_frequency, 1 - allele_1_frequency)


def bin_maf(maf_values, rare_cutoff, common_cutoff):
    """
    Label each SNP as rare (MAF < rare_cutoff), low frequency
    (rare_cutoff <= MAF <= common_cutoff) or common (MAF > common_cutoff).
    """
    return np.where(
        maf_values < rare_cutoff, "rare",
        np.where(maf_values > common_cutoff, "common", "low_frequency"),
    )


def maf_summary(genotype_dir, file_prefix, chromosomes, rare_cutoff, common_cutoff):
    """Per-chromosome counts of rare, low frequency and common SNPs, plus a genome-wide total row."""
    rows = []
    for chromosome in chromosomes:
        dosages = read_genotypes(genotype_dir, file_prefix, chromosome)
        labels = bin_maf(minor_allele_frequency(dosages), rare_cutoff, common_cutoff)
        counts = pd.Series(labels).value_counts()
        rows.append({
            "chromosome": str(chromosome),
            "rare": int(counts.get("rare", 0)),
            "low_frequency": int(counts.get("low_frequency", 0)),
            "common": int(counts.get("common", 0)),
        })
    summary = pd.DataFrame(rows)
    total = summary[["rare", "low_frequency", "common"]].sum()
    summary.loc[len(summary)] = {"chromosome": "all", **total.to_dict()}
    summary["total_snps"] = summary[["rare", "low_frequency", "common"]].sum(axis=1)
    for category in ["rare", "low_frequency", "common"]:
        summary[f"{category}_proportion"] = (summary[category] / summary["total_snps"]).round(4)
    return summary


def describe_files(genotype_dir, file_prefix, example_chromosome, expression_path):
    """Plain-text description of both data sources, for the week 1 questions."""
    bim = read_bim(genotype_dir, file_prefix, example_chromosome)
    fam = read_fam(genotype_dir, file_prefix, example_chromosome)
    annotation, expression = read_expression(expression_path)
    extensions = sorted({os.path.splitext(name)[1] for name in os.listdir(genotype_dir)})

    lines = [
        "GENOTYPE DATA (1000 Genomes, LDREF)",
        f"  Folder: {genotype_dir}",
        f"  File extensions: {', '.join(extensions)} (PLINK binary format, one set per chromosome)",
        "  .bed = binary genotype calls (not human-readable)",
        "  .bim = one row per variant, fields: " + ", ".join(bim.columns),
        "  .fam = one row per individual, fields: " + ", ".join(fam.columns),
        f"  Chromosome {example_chromosome}: {len(bim):,} SNPs, {len(fam):,} individuals",
        "",
        "  First rows of .bim:",
        bim.head(3).to_string(index=False),
        "",
        "  First rows of .fam:",
        fam.head(3).to_string(index=False),
        "",
        "EXPRESSION DATA (Geuvadis LCL RNA-seq)",
        f"  File: {expression_path}",
        f"  File extension: {os.path.splitext(expression_path)[1] or '(none)'}; tab-separated text",
        "  Annotation fields: " + ", ".join(["TargetID"] + list(annotation.columns)),
        f"  {expression.shape[0]:,} genes x {expression.shape[1]:,} individuals",
        "",
        "SOFTWARE",
        "  PLINK / plink2 is the standard tool for .bed/.bim/.fam files.",
        "  In code: plink2R (R), bed-reader or pandas-plink (Python).",
    ]
    return "\n".join(lines)


def sample_overlap(genotype_dir, file_prefix, example_chromosome, expression_path):
    """Counts of individuals in each dataset and in both."""
    fam = read_fam(genotype_dir, file_prefix, example_chromosome)
    _, expression = read_expression(expression_path)
    shared = shared_sample_ids(fam, expression)
    return pd.DataFrame([{
        "genotyped_individuals": len(fam),
        "expression_individuals": expression.shape[1],
        "individuals_in_both": len(shared),
    }])


def snps_per_cis_window(genotype_dir, file_prefix, chromosomes, expression_path, window_bp):
    """
    Number of genotyped SNPs within +/- window_bp of every autosomal gene in the
    expression data (the cis window used for eQTL mapping). Returns the
    per-gene counts and a one-row summary of their distribution.
    """
    annotation, _ = read_expression(expression_path)
    rows = []
    for chromosome in chromosomes:
        positions = np.sort(read_bim(genotype_dir, file_prefix, chromosome)["position_bp"].to_numpy())
        genes = annotation[annotation["Chr"] == str(chromosome)]
        starts = np.searchsorted(positions, genes["Coord"].to_numpy() - window_bp, side="left")
        ends = np.searchsorted(positions, genes["Coord"].to_numpy() + window_bp, side="right")
        rows.append(pd.DataFrame({
            "target_id": genes.index,
            "chromosome": str(chromosome),
            "gene_coordinate": genes["Coord"].to_numpy(),
            "n_cis_snps": ends - starts,
        }))
    counts = pd.concat(rows, ignore_index=True)
    described = counts["n_cis_snps"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    summary = pd.DataFrame([{
        "n_genes": int(described["count"]),
        "mean": round(described["mean"], 1),
        "median": described["50%"],
        "p05": described["5%"],
        "p25": described["25%"],
        "p75": described["75%"],
        "p95": described["95%"],
        "min": described["min"],
        "max": described["max"],
        "genes_with_zero_snps": int((counts["n_cis_snps"] == 0).sum()),
    }])
    return counts, summary
