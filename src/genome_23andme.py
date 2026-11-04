"""
genome_23andme.py: read a 23andMe raw data file and line it up with the
1000 Genomes panel (week 5).

A 23andMe raw file is tab-separated text with comment lines starting with
"#" and four columns: rsid, chromosome, position, genotype (e.g. "AG",
"--" for a failed call). Positions are on build 37 (hg19), the same build as
the LDREF panel.
"""

import os

import numpy as np
import pandas as pd
from bed_reader import to_bed

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def read_23andme(path):
    """Raw 23andMe calls as a table with rsid, chromosome, position, genotype."""
    calls = pd.read_csv(
        path, sep="\t", comment="#", header=None,
        names=["rsid", "chromosome", "position", "genotype"],
        dtype={"rsid": str, "chromosome": str, "position": "Int64", "genotype": str},
        compression="infer",
    )
    calls["genotype"] = calls["genotype"].fillna("--").str.upper()
    return calls


def dosages_for_panel(calls, panel_snps):
    """
    Copies of the panel's allele_1 at each panel SNP the person was genotyped at.

    panel_snps: table with snp_id, allele_1, allele_2 (from the .bim files).
    Calls are matched by rsID. A genotype is used only if both letters are
    the panel's two alleles (or both are on the opposite strand). Failed
    calls ("--"), indels ("I"/"D") and single-letter calls (X/Y/MT) are dropped.
    Returns a table with snp_id and dosage.
    """
    merged = calls.merge(panel_snps, left_on="rsid", right_on="snp_id", how="inner")
    merged = merged[merged["genotype"].str.fullmatch(r"[ACGT]{2}")]

    def dosage(row):
        alleles = {row["allele_1"], row["allele_2"]}
        letters = list(row["genotype"])
        if not set(letters) <= alleles:
            letters = [letter.translate(COMPLEMENT) for letter in letters]
            if not set(letters) <= alleles:
                return np.nan
        return float(sum(letter == row["allele_1"] for letter in letters))

    merged["dosage"] = merged.apply(dosage, axis=1)
    merged = merged.dropna(subset=["dosage"])
    return merged[["snp_id", "dosage"]].reset_index(drop=True)


def write_plink(calls_aligned, panel_snps, person_id, out_prefix):
    """
    Save the person's matched genotypes as PLINK .bed/.bim/.fam, using the
    panel's allele coding so the files can be merged with 1000 Genomes.
    """
    table = calls_aligned.merge(panel_snps, on="snp_id").sort_values(["chrom", "position_bp"])
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    to_bed(
        out_prefix + ".bed",
        table["dosage"].to_numpy(dtype="float32").reshape(1, -1),
        properties={
            "fid": [person_id], "iid": [person_id],
            "chromosome": table["chrom"].astype(str).tolist(),
            "sid": table["snp_id"].tolist(),
            "cm_position": table["genetic_distance_cm"].tolist(),
            "bp_position": table["position_bp"].tolist(),
            "allele_1": table["allele_1"].tolist(),
            "allele_2": table["allele_2"].tolist(),
        },
    )
