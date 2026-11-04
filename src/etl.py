"""
etl.py: loading and matching the 1000 Genomes genotype data and the
Geuvadis gene expression data.

Genotypes are PLINK binary files (.bed/.bim/.fam), one set per chromosome,
named like `1000G.EUR.15.bed`. Expression is a tab-separated table with one
row per gene and one column per individual.
"""

import os

import numpy as np
import pandas as pd
from bed_reader import open_bed

BIM_COLUMNS = ["chrom", "snp_id", "genetic_distance_cm", "position_bp", "allele_1", "allele_2"]
FAM_COLUMNS = ["family_id", "individual_id", "father_id", "mother_id", "sex", "phenotype"]
EXPRESSION_ANNOTATION_COLUMNS = ["TargetID", "Gene_Symbol", "Chr", "Coord"]


def genotype_prefix(genotype_dir, file_prefix, chromosome):
    """Path prefix (without extension) for one chromosome's PLINK files."""
    return os.path.join(genotype_dir, f"{file_prefix}.{chromosome}")


def read_bim(genotype_dir, file_prefix, chromosome):
    """Variant table for one chromosome: one row per SNP."""
    path = genotype_prefix(genotype_dir, file_prefix, chromosome) + ".bim"
    return pd.read_csv(path, sep=r"\s+", header=None, names=BIM_COLUMNS)


def read_fam(genotype_dir, file_prefix, chromosome):
    """Sample table for one chromosome: one row per individual."""
    path = genotype_prefix(genotype_dir, file_prefix, chromosome) + ".fam"
    return pd.read_csv(path, sep=r"\s+", header=None, names=FAM_COLUMNS, dtype={"individual_id": str})


def read_expression(expression_path):
    """
    Gene expression table. Returns (annotation, expression):
    annotation has one row per gene (TargetID, Gene_Symbol, Chr, Coord);
    expression has genes as rows and individual IDs as columns.
    """
    table = pd.read_csv(expression_path, sep="\t")
    table["Chr"] = table["Chr"].astype(str)
    annotation = table[EXPRESSION_ANNOTATION_COLUMNS].set_index("TargetID")
    expression = table.drop(columns=["Gene_Symbol", "Chr", "Coord"]).set_index("TargetID")
    return annotation, expression


def shared_sample_ids(fam_table, expression):
    """Individuals that have both genotypes and expression, in genotype file order."""
    expression_ids = set(expression.columns)
    return [sample_id for sample_id in fam_table["individual_id"] if sample_id in expression_ids]


def read_genotypes(genotype_dir, file_prefix, chromosome, snp_mask=None, sample_ids=None):
    """
    Genotype dosage matrix (individuals x SNPs) for one chromosome, counting
    copies of allele_1 (0, 1 or 2). Missing calls are filled with the SNP's
    mean dosage, matching plink2R's impute="avg".

    snp_mask: optional boolean array selecting SNPs (e.g. a cis window).
    sample_ids: optional list of individual IDs to keep, in the order given.
    """
    bed_path = genotype_prefix(genotype_dir, file_prefix, chromosome) + ".bed"
    with open_bed(bed_path) as bed:
        all_ids = np.asarray(bed.iid).astype(str)
        if sample_ids is None:
            sample_index = np.arange(len(all_ids))
        else:
            position_of = {sample_id: i for i, sample_id in enumerate(all_ids)}
            sample_index = np.array([position_of[s] for s in sample_ids])
        snp_index = np.flatnonzero(snp_mask) if snp_mask is not None else None
        dosages = bed.read(index=np.s_[sample_index, snp_index], dtype="float32")

    snp_means = np.nanmean(dosages, axis=0)
    missing_rows, missing_cols = np.where(np.isnan(dosages))
    dosages[missing_rows, missing_cols] = snp_means[missing_cols]
    return dosages
