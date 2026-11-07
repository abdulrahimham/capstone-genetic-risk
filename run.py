#!/usr/bin/env python
"""
Build script for the DSC 180A Quarter 1 Project (genetic risk prediction).

Usage:
    python run.py data       # check raw data is in place, write file descriptions and sample overlap
    python run.py explore    # MAF bins (rare / low frequency / common) per chromosome
    python run.py eqtl       # single-gene cis-eQTL scan + LocusZoom-style plot
    python run.py gene-prs   # week 4: cis-eQTL for every gene on a chromosome, gene expression PRS, cross-validation
    python run.py prs        # height PRS (clumping + thresholding) for 1000 Genomes and the 23andMe sample
    python run.py all        # data, explore, eqtl, gene-prs, prs on the real data
    python run.py test       # the full pipeline on the small made-up data in test/testdata
    python run.py clean      # delete everything in outputs/ and test/outputs/

Targets can be combined, e.g. `python run.py data eqtl`.
"""

import json
import logging
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from etl import read_expression, read_fam, read_genotypes, read_bim, shared_sample_ids  # noqa: E402
from explore import describe_files, maf_summary, sample_overlap, snps_per_cis_window  # noqa: E402
from eqtl import cis_eqtl_single_gene, lookup_gene, snps_tied_with_lead, summarize_top_hits  # noqa: E402
from plots import (cv_correlation_histogram, genotype_boxplot, locuszoom_plot,  # noqa: E402
                   prs_distribution_plot, threshold_r2_plot)
from gene_prs import (chromosome_cis_eqtl, cross_validate_gene, fit_gene_score,  # noqa: E402
                      gene_windows, split_samples, threshold_table)
from prs import place_in_distribution, qc_summary_statistics, score_panel, select_prs_snps  # noqa: E402
from genome_23andme import dosages_for_panel, read_23andme, write_plink  # noqa: E402

import pandas as pd  # noqa: E402

REAL_DATA_TARGETS = ["data", "explore", "eqtl", "gene-prs", "prs"]


def load_config(path):
    with open(path) as config_file:
        return json.load(config_file)


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(os.path.join(output_dir, "run.log")), logging.StreamHandler()],
        force=True,
    )


def check_inputs(data_params):
    """Fail early with a clear message if the raw data is not where the config says."""
    missing = []
    if not os.path.isdir(data_params["genotype_dir"]):
        missing.append(data_params["genotype_dir"])
    if not os.path.isfile(data_params["expression_path"]):
        missing.append(data_params["expression_path"])
    if missing:
        raise FileNotFoundError(
            "Missing input data: " + ", ".join(missing)
            + ". See README.md for where to download it."
        )


def run_data(data_params):
    check_inputs(data_params)
    output_dir = data_params["output_dir"]
    description = describe_files(
        data_params["genotype_dir"], data_params["genotype_file_prefix"],
        data_params["example_chromosome"], data_params["expression_path"],
    )
    with open(os.path.join(output_dir, "file_description.txt"), "w") as out:
        out.write(description + "\n")

    overlap = sample_overlap(
        data_params["genotype_dir"], data_params["genotype_file_prefix"],
        data_params["example_chromosome"], data_params["expression_path"],
    )
    overlap.to_csv(os.path.join(output_dir, "sample_overlap.csv"), index=False)
    logging.info("Sample overlap: %s", overlap.to_dict("records")[0])


def run_explore(data_params, explore_params):
    check_inputs(data_params)
    summary = maf_summary(
        data_params["genotype_dir"], data_params["genotype_file_prefix"], data_params["chromosomes"],
        explore_params["rare_maf_cutoff"], explore_params["common_maf_cutoff"],
    )
    summary.to_csv(os.path.join(data_params["output_dir"], "maf_summary.csv"), index=False)
    logging.info("Genome-wide MAF bins: %s", summary.iloc[-1].to_dict())

    window_counts, window_summary = snps_per_cis_window(
        data_params["genotype_dir"], data_params["genotype_file_prefix"], data_params["chromosomes"],
        data_params["expression_path"], explore_params["cis_window_bp"],
    )
    window_counts.to_csv(os.path.join(data_params["output_dir"], "snps_per_cis_window.csv"), index=False)
    window_summary.to_csv(os.path.join(data_params["output_dir"], "snps_per_cis_window_summary.csv"), index=False)
    logging.info("SNPs per cis window: %s", window_summary.to_dict("records")[0])


def run_eqtl(data_params, eqtl_params):
    check_inputs(data_params)
    output_dir = data_params["output_dir"]
    genotype_dir = data_params["genotype_dir"]
    file_prefix = data_params["genotype_file_prefix"]
    gene_label = eqtl_params["gene_label"]

    annotation, expression = read_expression(data_params["expression_path"])
    results, gene_info = cis_eqtl_single_gene(
        genotype_dir, file_prefix, annotation, expression,
        eqtl_params["gene_id"], eqtl_params["cis_window_bp"],
    )
    top, bonferroni_threshold, n_significant = summarize_top_hits(results, gene_info, eqtl_params["n_top_snps"])
    gene_info.update({
        "gene_label": gene_label,
        "bonferroni_threshold": bonferroni_threshold,
        "n_significant_snps": n_significant,
        "lead_snp": top.iloc[0]["snp_id"],
        "lead_snp_p_value": float(top.iloc[0]["p_value"]),
        "lead_snp_beta": float(top.iloc[0]["beta"]),
        "lead_snp_r_squared": float(top.iloc[0]["r_squared"]),
        "snps_tied_with_lead": snps_tied_with_lead(results),
    })

    results.to_csv(os.path.join(output_dir, f"cis_eqtl_{gene_label}.csv"), index=False)
    top.to_csv(os.path.join(output_dir, f"cis_eqtl_{gene_label}_top_snps.csv"), index=False)
    with open(os.path.join(output_dir, f"cis_eqtl_{gene_label}_summary.json"), "w") as out:
        json.dump(gene_info, out, indent=2, default=str)

    locuszoom_plot(results, gene_info, bonferroni_threshold, gene_label,
                   os.path.join(output_dir, f"locuszoom_{gene_label}.png"))

    # Expression by genotype for the lead SNP
    chromosome = gene_info["chromosome"]
    bim = read_bim(genotype_dir, file_prefix, chromosome)
    lead_mask = (bim["snp_id"] == gene_info["lead_snp"]).to_numpy()
    sample_ids = shared_sample_ids(read_fam(genotype_dir, file_prefix, chromosome), expression)
    lead_dosage = read_genotypes(genotype_dir, file_prefix, chromosome, snp_mask=lead_mask, sample_ids=sample_ids)[:, 0]
    target_id, _ = lookup_gene(annotation, eqtl_params["gene_id"])
    genotype_boxplot(np.rint(lead_dosage), expression.loc[target_id, sample_ids].to_numpy(dtype=float),
                     gene_info["lead_snp"], gene_label,
                     os.path.join(output_dir, f"lead_snp_boxplot_{gene_label}.png"))

    logging.info("cis-eQTL %s: %s", gene_label, {k: gene_info[k] for k in
                 ["n_tested_snps", "n_significant_snps", "lead_snp", "lead_snp_p_value"]})


def run_gene_prs(data_params, gene_params):
    """
    Week 4.
    Task 1: cis-eQTL for every gene on one chromosome (all shared individuals).
    Task 2: C+T expression score for one gene, R^2 at each threshold, fit and
            evaluated on the same people (shows overfitting).
    Task 3: 80/10/10 cross-validation for that gene, then the same for every
            gene on the chromosome to get the distribution of prediction accuracy.
    """
    check_inputs(data_params)
    output_dir = data_params["output_dir"]
    genotype_dir = data_params["genotype_dir"]
    file_prefix = data_params["genotype_file_prefix"]
    chromosome = gene_params["chromosome"]
    gene_label = gene_params["example_gene_label"]

    annotation, expression = read_expression(data_params["expression_path"])
    bim = read_bim(genotype_dir, file_prefix, chromosome)
    sample_ids = shared_sample_ids(read_fam(genotype_dir, file_prefix, chromosome), expression)
    dosages = read_genotypes(genotype_dir, file_prefix, chromosome, sample_ids=sample_ids)
    positions = bim["position_bp"].to_numpy()
    snp_ids = bim["snp_id"].to_numpy()
    windows = gene_windows(annotation, chromosome, bim, gene_params["cis_window_bp"])
    windows = {gene: columns for gene, columns in windows.items() if len(columns) > 0}
    expression_shared = expression.loc[list(windows), sample_ids]
    gene_symbols = annotation["Gene_Symbol"].to_dict()

    # Task 1
    all_tests, per_gene = chromosome_cis_eqtl(dosages, positions, snp_ids, expression_shared, windows, gene_symbols)
    all_tests.to_csv(os.path.join(output_dir, f"chr{chromosome}_cis_eqtl_all_tests.csv.gz"), index=False)
    per_gene.to_csv(os.path.join(output_dir, f"chr{chromosome}_cis_eqtl_by_gene.csv"), index=False)
    logging.info("chr%s cis-eQTL: %d genes, %d with a Bonferroni-significant eQTL",
                 chromosome, len(per_gene), int(per_gene["is_egene_bonferroni"].sum()))

    # Task 2 (example gene, all people)
    from eqtl import lookup_gene
    example_id, _ = lookup_gene(annotation, gene_params["example_gene_id"])
    example_columns = windows[example_id]
    example_dosages = dosages[:, example_columns]
    example_expression = expression_shared.loc[example_id].to_numpy(dtype=float)
    clumped_all = fit_gene_score(example_dosages, example_expression, positions[example_columns],
                                 gene_params["clump_kb"], gene_params["clump_r2"])
    full_sample = threshold_table(example_dosages, example_expression, clumped_all, gene_params["p_thresholds"])
    full_sample.to_csv(os.path.join(output_dir, f"gene_prs_{gene_label}_in_sample.csv"), index=False)

    # Task 3 (cross-validation)
    train_ids, tune_ids, test_ids = split_samples(sample_ids, gene_params["train_fraction"],
                                                  gene_params["tune_fraction"], gene_params["random_seed"])
    row_of = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    group_rows = {name: np.array([row_of[s] for s in ids])
                  for name, ids in [("train", train_ids), ("tune", tune_ids), ("test", test_ids)]}

    def cv_for(target_id):
        columns = windows[target_id]
        values = expression_shared.loc[target_id].to_numpy(dtype=float)
        return cross_validate_gene(
            {g: dosages[rows][:, columns] for g, rows in group_rows.items()},
            {g: values[rows] for g, rows in group_rows.items()},
            positions[columns], gene_params["p_thresholds"], gene_params["clump_kb"], gene_params["clump_r2"],
        )

    example_cv, example_tuning = cv_for(example_id)
    example_tuning.to_csv(os.path.join(output_dir, f"gene_prs_{gene_label}_cv_tuning.csv"), index=False)
    example_cv.update({"gene": gene_label, "n_train": len(train_ids), "n_tune": len(tune_ids), "n_test": len(test_ids),
                       "in_sample_r_squared_by_threshold": dict(zip(full_sample["p_threshold"].astype(str),
                                                                    full_sample["r_squared"].round(4)))})
    with open(os.path.join(output_dir, f"gene_prs_{gene_label}_cv_summary.json"), "w") as out:
        json.dump(example_cv, out, indent=2)
    threshold_r2_plot(full_sample, example_tuning, gene_label,
                      os.path.join(output_dir, f"gene_prs_{gene_label}_thresholds.png"))
    logging.info("%s cross-validation: %s", gene_label, example_cv)

    cv_rows = []
    for target_id in windows:
        summary, _ = cv_for(target_id)
        cv_rows.append({"target_id": target_id, **summary})
    cv_table = pd.DataFrame(cv_rows).merge(per_gene[["target_id", "gene_symbol", "is_egene_bonferroni"]], on="target_id")
    cv_table.to_csv(os.path.join(output_dir, f"chr{chromosome}_cv_by_gene.csv"), index=False)

    def describe(values):
        values = values.dropna()
        return {"n_genes": int(len(values)), "mean": round(float(values.mean()), 3),
                "median": round(float(values.median()), 3),
                "q25": round(float(values.quantile(0.25)), 3), "q75": round(float(values.quantile(0.75)), 3),
                "share_positive": round(float((values > 0).mean()), 3)}

    distribution = {
        "chromosome": chromosome,
        "n_test_individuals": len(test_ids),
        "test_r_all_genes": describe(cv_table["test_r"]),
        "test_r_egenes": describe(cv_table.loc[cv_table["is_egene_bonferroni"], "test_r"]),
        "test_r_non_egenes": describe(cv_table.loc[~cv_table["is_egene_bonferroni"], "test_r"]),
        "test_r_squared_all_genes": describe(cv_table["test_r_squared"]),
        "genes_with_constant_prediction": int(cv_table["test_r"].isna().sum()),
        "best_threshold_counts": cv_table["best_p_threshold"].value_counts().sort_index().to_dict(),
    }
    distribution["best_threshold_counts"] = {str(k): int(v) for k, v in distribution["best_threshold_counts"].items()}
    with open(os.path.join(output_dir, f"chr{chromosome}_cv_distribution.json"), "w") as out:
        json.dump(distribution, out, indent=2)
    cv_correlation_histogram(cv_table, chromosome, os.path.join(output_dir, f"chr{chromosome}_cv_correlation.png"))
    logging.info("chr%s CV distribution: %s", chromosome, distribution["test_r_all_genes"])


def run_prs(data_params, prs_params):
    """
    Week 5: C+T polygenic score for every 1000 Genomes individual and, if a
    23andMe file is available, for that person too. When the person is
    included, only SNPs they were genotyped at are used, so everyone is
    scored on exactly the same SNPs.
    """
    check_inputs(data_params)
    output_dir = data_params["output_dir"]
    genotype_dir = data_params["genotype_dir"]
    file_prefix = data_params["genotype_file_prefix"]
    chromosomes = data_params["chromosomes"]
    label = prs_params["trait_label"]
    file_label = label.lower().replace(" ", "_")

    gwas = pd.read_csv(prs_params["gwas_path"], sep=r"\s+")
    gwas, counts = qc_summary_statistics(gwas, prs_params["min_info"], prs_params["min_maf"])

    person_path = prs_params.get("person_23andme_path")
    person_label = prs_params.get("person_label", "person")
    person_dosages = None
    if person_path and os.path.isfile(person_path):
        panel_snps = pd.concat([read_bim(genotype_dir, file_prefix, c) for c in chromosomes], ignore_index=True)
        calls = read_23andme(person_path)
        person_dosages = dosages_for_panel(calls, panel_snps)
        write_plink(person_dosages, panel_snps, person_label,
                    os.path.join(output_dir, f"{person_label.lower()}_plink", person_label.lower()))
        counts.update({"person_calls_in_file": len(calls), "person_calls_matched_to_panel": len(person_dosages)})
        keep_snps = set(person_dosages["snp_id"])
    else:
        logging.warning("No 23andMe file at %s; scoring the 1000 Genomes panel only.", person_path)
        keep_snps = None

    selected, selection_counts = select_prs_snps(
        gwas, genotype_dir, file_prefix, chromosomes,
        prs_params["p_threshold"], prs_params["clump_kb"], prs_params["clump_r2"], keep_snps=keep_snps,
    )
    counts.update(selection_counts)
    selected.to_csv(os.path.join(output_dir, f"prs_{file_label}_snps.csv"), index=False)

    panel_scores = score_panel(selected, genotype_dir, file_prefix)
    panel_scores.to_csv(os.path.join(output_dir, f"prs_{file_label}_1000G.csv"), index=False)

    summary = {"trait": label, **{k: prs_params[k] for k in ["p_threshold", "clump_kb", "clump_r2"]}, "snp_counts": counts,
               "panel_mean": float(panel_scores["prs"].mean()), "panel_sd": float(panel_scores["prs"].std(ddof=1))}
    person_result = None
    if person_dosages is not None:
        weights = selected.set_index("SNP")["beta_panel"]
        person_score = float((person_dosages.set_index("snp_id")["dosage"].reindex(weights.index) * weights).sum())
        person_result = place_in_distribution(person_score, panel_scores["prs"].to_numpy())
        summary[person_label] = person_result

    with open(os.path.join(output_dir, f"prs_{file_label}_summary.json"), "w") as out:
        json.dump(summary, out, indent=2)

    prs_distribution_plot(
        panel_scores["prs"].to_numpy(), label, os.path.join(output_dir, f"prs_{file_label}_distribution.png"),
        person_score=person_result["prs"] if person_result else None,
        person_label=person_label, percentile=person_result["percentile"] if person_result else None,
    )
    logging.info("PRS %s: %s", label, {"snps_in_score": counts["after_clumping"], label and person_label: person_result})


def run_clean():
    for folder in ["outputs", os.path.join("test", "outputs")]:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"Removed {folder}/")


def main(targets):
    if not targets:
        print(__doc__)
        return
    if "clean" in targets:
        run_clean()
        return

    if "test" in targets:
        data_params = load_config("config/test-params.json")
        eqtl_params = load_config("config/test-eqtl-params.json")
        prs_params = load_config("config/test-prs-params.json")
        gene_params = load_config("config/test-gene-prs-params.json")
        targets = REAL_DATA_TARGETS
    else:
        data_params = load_config("config/data-params.json")
        eqtl_params = load_config("config/eqtl-params.json")
        prs_params = load_config("config/prs-params.json")
        gene_params = load_config("config/gene-prs-params.json")
        if "all" in targets:
            targets = REAL_DATA_TARGETS
    explore_params = load_config("config/explore-params.json")

    setup_logging(data_params["output_dir"])
    if "data" in targets:
        run_data(data_params)
    if "explore" in targets:
        run_explore(data_params, explore_params)
    if "eqtl" in targets:
        run_eqtl(data_params, eqtl_params)
    if "gene-prs" in targets:
        run_gene_prs(data_params, gene_params)
    if "prs" in targets:
        run_prs(data_params, prs_params)


if __name__ == "__main__":
    main(sys.argv[1:])
