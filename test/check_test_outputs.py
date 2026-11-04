"""
Checks the outputs of `python run.py test` against the answers worked out
by hand in test/make_test_data.py. Exits with an error if anything is off.
"""

import json
import os
import sys

import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    failures = []

    overlap = pd.read_csv(os.path.join(OUTPUT_DIR, "sample_overlap.csv")).iloc[0]
    if (overlap["genotyped_individuals"], overlap["expression_individuals"], overlap["individuals_in_both"]) != (10, 10, 9):
        failures.append(f"sample overlap was {overlap.to_dict()}, expected 10 / 10 / 9")

    maf = pd.read_csv(os.path.join(OUTPUT_DIR, "maf_summary.csv")).set_index("chromosome").loc["all"]
    if (maf["rare"], maf["low_frequency"], maf["common"]) != (1, 1, 4):
        failures.append(f"MAF bins were {maf[['rare', 'low_frequency', 'common']].to_dict()}, expected 1 / 1 / 4")

    with open(os.path.join(OUTPUT_DIR, "cis_eqtl_TESTGENE1_summary.json")) as summary_file:
        summary = json.load(summary_file)
    if summary["n_cis_snps"] != 5:
        failures.append(f"n_cis_snps was {summary['n_cis_snps']}, expected 5 (rs_far must be excluded)")
    if summary["n_tested_snps"] != 4:
        failures.append(f"n_tested_snps was {summary['n_tested_snps']}, expected 4 (rs_monomorph untestable)")
    if summary["lead_snp"] != "rs_signal":
        failures.append(f"lead SNP was {summary['lead_snp']}, expected rs_signal")
    if abs(summary["lead_snp_beta"] - 2.0) > 1e-6:
        failures.append(f"lead SNP beta was {summary['lead_snp_beta']}, expected 2.0")

    results = pd.read_csv(os.path.join(OUTPUT_DIR, "cis_eqtl_TESTGENE1.csv"))
    if pd.isna(results.set_index("snp_id").loc["rs_missing", "p_value"]):
        failures.append("rs_missing got no p value; the missing call should be mean-filled")
    if "rs_far" in set(results["snp_id"]):
        failures.append("rs_far (outside the cis window) was tested")

    for name in ["locuszoom_TESTGENE1.png", "lead_snp_boxplot_TESTGENE1.png", "file_description.txt"]:
        if not os.path.isfile(os.path.join(OUTPUT_DIR, name)):
            failures.append(f"missing output {name}")

    # Week 4: chromosome scan and cross-validation ran
    by_gene = pd.read_csv(os.path.join(OUTPUT_DIR, "chr1_cis_eqtl_by_gene.csv")).set_index("target_id")
    if by_gene.loc["TESTGENE1", "lead_snp"] != "rs_signal" or by_gene.loc["TESTGENE1", "n_cis_snps"] != 5:
        failures.append(f"chromosome scan for TESTGENE1 was {by_gene.loc['TESTGENE1'].to_dict()}")
    for name in ["chr1_cv_by_gene.csv", "chr1_cv_distribution.json", "gene_prs_TESTGENE1_cv_summary.json"]:
        if not os.path.isfile(os.path.join(OUTPUT_DIR, name)):
            failures.append(f"missing output {name}")

    # Week 5: PRS
    with open(os.path.join(OUTPUT_DIR, "prs_test_trait_summary.json")) as prs_file:
        prs_summary = json.load(prs_file)
    snp_counts = prs_summary["snp_counts"]
    if snp_counts["gwas_snps_after_qc"] != 3:
        failures.append(f"{snp_counts['gwas_snps_after_qc']} GWAS SNPs passed QC, expected 3")
    if snp_counts["person_calls_matched_to_panel"] != 3:
        failures.append(f"{snp_counts['person_calls_matched_to_panel']} 23andMe calls matched, expected 3")
    if snp_counts["after_clumping"] != 2:
        failures.append(f"{snp_counts['after_clumping']} SNPs in the score, expected 2")
    person = prs_summary["TESTPERSON"]
    if abs(person["prs"] - 0.8) > 1e-6 or person["percentile"] != 100:
        failures.append(f"TESTPERSON was {person}, expected prs 0.8 at the 100th percentile")
    panel = pd.read_csv(os.path.join(OUTPUT_DIR, "prs_test_trait_1000G.csv"))
    if abs(panel["prs"].max() - 0.6) > 1e-6:
        failures.append(f"max panel score was {panel['prs'].max()}, expected 0.6")

    # Clumping and allele alignment, checked directly
    sys.path.insert(0, os.path.join(os.path.dirname(OUTPUT_DIR), "..", "src"))
    import numpy as np
    from prs import align_to_panel, clump_chromosome
    same_genotypes = np.array([0, 1, 2, 1, 0, 2], dtype=float)
    snps = pd.DataFrame({"position_bp": [0, 100_000, 300_000], "P": [1e-5, 1e-4, 1e-3]})
    kept = clump_chromosome(snps, np.column_stack([same_genotypes] * 3), clump_kb=250, clump_r2=0.1)
    if list(kept) != [True, False, True]:
        failures.append(f"clumping kept {list(kept)}, expected [True, False, True]")
    gwas_rows = pd.DataFrame({"SNP": ["s1", "s2", "s3"], "A1": ["A", "G", "T"], "A2": ["G", "A", "C"], "BETA": [0.5, 0.5, 0.5]})
    bim_rows = pd.DataFrame({"snp_id": ["s1", "s2", "s3"], "allele_1": ["A", "A", "A"], "allele_2": ["G", "G", "G"]})
    aligned = align_to_panel(gwas_rows, bim_rows).set_index("SNP")["beta_panel"].to_dict()
    if aligned != {"s1": 0.5, "s2": -0.5, "s3": 0.5}:
        failures.append(f"allele alignment gave {aligned}, expected s1 0.5, s2 -0.5 (swapped), s3 0.5 (other strand)")

    if failures:
        print("TEST FAILED")
        for failure in failures:
            print("  -", failure)
        sys.exit(1)
    print("All test checks passed.")


if __name__ == "__main__":
    main()
