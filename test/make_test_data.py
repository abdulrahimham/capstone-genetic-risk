"""
Builds the small made-up dataset in test/testdata/ used by `python run.py test`.

This is unit test data, not a sample of the real data. Every row is chosen
to check one behavior, and the right answers can be worked out by hand.

Individuals
- 10 genotyped (TEST01..TEST10). TEST10 has no expression.
- Expression has TEST01..TEST09 plus TEST99, who is not genotyped.
  -> expected overlap: 9 individuals in both.

Gene
- TESTGENE1 on chromosome 1 at position 1,000,000. cis window = 500,000 to 1,500,000.

SNPs (chromosome 1)
- rs_signal     at   900,000: expression = 1 + 2 * dosage + small noise that is
                              uncorrelated with dosage -> expected beta = 2.0, the lead SNP.
- rs_null       at 1,200,000: dosage pattern unrelated to expression -> beta near 0.
- rs_missing    at 1,100,000: an unrelated dosage pattern where TEST05's call is
                              missing; it should be filled with the SNP mean, so the
                              SNP still gets a p value instead of being dropped.
- rs_monomorph  at 1,300,000: everyone has dosage 0 -> MAF 0 (rare), no regression (NaN).
- rs_far        at 2,000,000: outside the cis window -> must not be tested.
- rs_lowfreq    at 1,400,000: one copy of the allele among 20 -> MAF 0.05 (low frequency).

Expected results (checked in test/check_test_outputs.py)
- sample overlap: 10 genotyped, 10 expression columns, 9 in both
- MAF bins: 1 rare, 1 low frequency, 4 common
- cis SNPs: 5 (everything except rs_far), tested SNPs: 4 (rs_monomorph is NaN)
- lead SNP: rs_signal with beta = 2.0

PRS test files (week 5)
- test_gwas.txt: rs_signal (A1=A, beta 0.5, p 1e-8), rs_far (A1=G, the panel's
  allele_2, beta 0.2, p 1e-5), rs_null (p 0.5), rs_monomorph (MAF 0.005, fails QC),
  rs_missing (INFO 0.5, fails QC).
- test_23andme.txt: TESTPERSON with rs_signal AA, rs_far AG, rs_missing AA,
  rs_null "--" (failed call).
Expected
- 3 GWAS SNPs pass QC; 3 of the person's calls match the panel
- score SNPs: rs_signal and rs_far (1.1 Mb apart, so clumping keeps both)
- rs_far's effect flips to -0.2 per copy of the panel's allele A
- person score = 2 x 0.5 + 1 x (-0.2) = 0.8
- panel scores = 0.5 x signal - 0.2 x far = 0.3 x signal dosage (max 0.6), so
  the person is at the 100th percentile
"""

import os

import numpy as np
import pandas as pd
from bed_reader import to_bed

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")

GENOTYPED_IDS = [f"TEST{i:02d}" for i in range(1, 11)]
EXPRESSION_IDS = GENOTYPED_IDS[:9] + ["TEST99"]

SIGNAL_DOSAGE = np.array([0, 0, 1, 1, 1, 2, 2, 2, 1, 0], dtype=float)
NULL_DOSAGE = np.array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1], dtype=float)


def noise_uncorrelated_with(dosage):
    """A fixed noise vector with mean 0 and zero correlation with `dosage`, so the slope stays exactly 2."""
    raw_noise = np.array([0.3, -0.2, 0.1, -0.4, 0.2, -0.1, 0.4, -0.3, 0.0])
    centered_dosage = dosage - dosage.mean()
    raw_noise = raw_noise - raw_noise.mean()
    projection = (raw_noise @ centered_dosage) / (centered_dosage @ centered_dosage)
    return raw_noise - projection * centered_dosage


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    missing_dosage = np.array([2, 1, 0, 0, 1, 1, 2, 0, 2, 1], dtype=float)
    missing_dosage[4] = np.nan  # TEST05
    monomorphic_dosage = np.zeros(10)
    far_dosage = SIGNAL_DOSAGE.copy()
    lowfreq_dosage = np.zeros(10)
    lowfreq_dosage[0] = 1

    snp_table = pd.DataFrame({
        "snp_id": ["rs_signal", "rs_missing", "rs_null", "rs_monomorph", "rs_lowfreq", "rs_far"],
        "position_bp": [900_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000, 2_000_000],
    })
    dosage_matrix = np.column_stack([
        SIGNAL_DOSAGE, missing_dosage, NULL_DOSAGE, monomorphic_dosage, lowfreq_dosage, far_dosage,
    ]).astype("float32")

    to_bed(
        os.path.join(OUTPUT_DIR, "test.1.bed"),
        dosage_matrix,
        properties={
            "fid": GENOTYPED_IDS,
            "iid": GENOTYPED_IDS,
            "chromosome": ["1"] * len(snp_table),
            "sid": snp_table["snp_id"].tolist(),
            "cm_position": [0.0] * len(snp_table),
            "bp_position": snp_table["position_bp"].tolist(),
            "allele_1": ["A"] * len(snp_table),
            "allele_2": ["G"] * len(snp_table),
        },
    )

    shared_signal = SIGNAL_DOSAGE[:9]
    expression_values = 1 + 2 * shared_signal + noise_uncorrelated_with(shared_signal)
    expression_row = dict(zip(EXPRESSION_IDS, list(np.round(expression_values, 10)) + [5.0]))
    other_gene_row = dict(zip(EXPRESSION_IDS, [0.5] * 10))

    expression = pd.DataFrame([
        {"TargetID": "TESTGENE1", "Gene_Symbol": "TESTGENE1", "Chr": 1, "Coord": 1_000_000, **expression_row},
        {"TargetID": "TESTGENE2", "Gene_Symbol": "TESTGENE2", "Chr": 2, "Coord": 5_000_000, **other_gene_row},
    ])
    expression.to_csv(os.path.join(OUTPUT_DIR, "test_expression.txt"), sep="\t", index=False)

    gwas = pd.DataFrame([
        [1, 900_000, "rs_signal", "A", "G", 1000, 0.01, 1e-8, np.exp(0.5), 0.9, 0.4],
        [1, 2_000_000, "rs_far", "G", "A", 1000, 0.01, 1e-5, np.exp(0.2), 0.9, 0.4],
        [1, 1_200_000, "rs_null", "A", "G", 1000, 0.01, 0.5, 1.1, 0.9, 0.4],
        [1, 1_300_000, "rs_monomorph", "A", "G", 1000, 0.01, 1e-9, 2.0, 0.9, 0.005],
        [1, 1_100_000, "rs_missing", "A", "G", 1000, 0.01, 1e-6, np.exp(0.1), 0.5, 0.4],
    ], columns=["CHR", "BP", "SNP", "A1", "A2", "N", "SE", "P", "OR", "INFO", "MAF"])
    gwas.to_csv(os.path.join(OUTPUT_DIR, "test_gwas.txt"), sep="\t", index=False)

    with open(os.path.join(OUTPUT_DIR, "test_23andme.txt"), "w") as out:
        out.write("# Made-up 23andMe-style file for unit tests\n# rsid\tchromosome\tposition\tgenotype\n")
        out.write("rs_signal\t1\t900000\tAA\n")
        out.write("rs_missing\t1\t1100000\tAA\n")
        out.write("rs_null\t1\t1200000\t--\n")
        out.write("rs_far\t1\t2000000\tAG\n")
        out.write("rs_not_in_panel\t1\t3000000\tCC\n")
    print(f"Wrote unit test data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
