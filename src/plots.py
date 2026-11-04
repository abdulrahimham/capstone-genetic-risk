"""
plots.py: figures for the report.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def locuszoom_plot(results, gene_info, bonferroni_threshold, gene_label, outpath):
    """
    LocusZoom-style plot: -log10(p) of each cis SNP against its position (Mb),
    with the gene's coordinate and the Bonferroni line marked and the lead SNP labeled.
    """
    tested = results.dropna(subset=["p_value"])
    lead = tested.loc[tested["p_value"].idxmin()]
    positions_mb = tested["position_bp"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(positions_mb, tested["neg_log10_p"], s=14, color="steelblue", alpha=0.75, linewidths=0)
    ax.scatter(lead["position_bp"] / 1e6, lead["neg_log10_p"], s=60, color="darkorange",
               edgecolors="white", linewidths=1.5, zorder=3)
    n_tied = len(gene_info.get("snps_tied_with_lead", [])) - 1
    lead_label = lead["snp_id"] + (f" (+{n_tied} tied, same genotypes)" if n_tied > 0 else "")
    ax.annotate(lead_label, (lead["position_bp"] / 1e6, lead["neg_log10_p"]),
                xytext=(10, 6), textcoords="offset points", va="bottom", fontsize=9, color="dimgray")

    ax.axhline(-np.log10(bonferroni_threshold), color="gray", linestyle="--", linewidth=1)
    ax.text(positions_mb.min(), -np.log10(bonferroni_threshold), " Bonferroni (0.05 / tested SNPs)",
            va="bottom", fontsize=8, color="gray")
    ax.axvline(gene_info["gene_coordinate"] / 1e6, color="lightgray", linewidth=1, zorder=0)

    ax.set_xlabel(f"Position on chromosome {gene_info['chromosome']} (Mb, hg19)")
    ax.set_ylabel("-log10(p value)")
    ax.set_title(f"cis-eQTL scan for {gene_label} "
                 f"({gene_info['n_tested_snps']:,} SNPs, {gene_info['n_individuals']} individuals)",
                 fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="whitesmoke")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def genotype_boxplot(dosages_rounded, expression_values, snp_id, gene_label, outpath):
    """Expression by genotype group (0, 1, 2 copies of allele 1) for the lead SNP."""
    groups = [expression_values[dosages_rounded == count] for count in (0, 1, 2)]
    labels = [f"{count}\n(n={len(group)})" for count, group in zip((0, 1, 2), groups)]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.boxplot(groups, tick_labels=labels, widths=0.5, showfliers=False,
               medianprops={"color": "darkorange", "linewidth": 2})
    for position, group in enumerate(groups, start=1):
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(group))
        ax.scatter(position + jitter, group, s=8, color="steelblue", alpha=0.5, linewidths=0)
    ax.set_xlabel(f"Copies of allele 1 at {snp_id}")
    ax.set_ylabel(f"{gene_label} expression (normalized)")
    ax.set_title(f"{gene_label} expression by {snp_id} genotype", fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def prs_distribution_plot(panel_scores, trait_label, outpath, person_score=None, person_label=None, percentile=None):
    """
    Histogram of PRS across the reference panel with a fitted bell curve,
    and an optional vertical line marking one person's score.
    """
    from scipy import stats

    mean, sd = panel_scores.mean(), panel_scores.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(panel_scores, bins=30, density=True, color="steelblue", alpha=0.6,
            edgecolor="white", linewidth=1, label=f"1000 Genomes EUR (n={len(panel_scores)})")
    x_values = np.linspace(min(panel_scores.min(), person_score if person_score is not None else np.inf) - sd,
                           max(panel_scores.max(), person_score if person_score is not None else -np.inf) + sd, 300)
    ax.plot(x_values, stats.norm.pdf(x_values, mean, sd), color="navy", linewidth=2, label="Normal fit")

    if person_score is not None:
        ax.axvline(person_score, color="darkorange", linewidth=2.5)
        note = f"{person_label}: {percentile:.0f}th percentile" if percentile is not None else person_label
        ax.text(person_score, ax.get_ylim()[1] * 0.95, f"  {note}", color="dimgray", fontsize=10, va="top",
                ha="left" if person_score < x_values.mean() else "right")

    ax.set_xlabel(f"{trait_label} polygenic risk score (sum of effect sizes)")
    ax.set_ylabel("Density")
    ax.set_title(f"{trait_label} PRS across 1000 Genomes Europeans", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def cv_correlation_histogram(cv_table, chromosome, outpath):
    """
    Distribution across genes of the test-set correlation between predicted and
    true expression, split by whether the gene had a significant cis-eQTL.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-1, 1, 41)
    groups = [
        (cv_table[~cv_table["is_egene_bonferroni"]], "No significant cis-eQTL", "lightsteelblue"),
        (cv_table[cv_table["is_egene_bonferroni"]], "Significant cis-eQTL (Bonferroni)", "steelblue"),
    ]
    values_by_group = [subset["test_r"].dropna() for subset, _, _ in groups]
    ax.hist(values_by_group, bins=bins, stacked=True, color=[color for _, _, color in groups],
            edgecolor="white", linewidth=1,
            label=[f"{label} (n={len(values)}, median r={values.median():.2f})"
                   for values, (_, label, _) in zip(values_by_group, groups)])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Correlation (r) between predicted and true expression, held-out test set")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Gene expression prediction accuracy across chromosome {chromosome} genes",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def threshold_r2_plot(full_sample, tuning, gene_label, outpath):
    """R^2 at each p-value threshold: in-sample (all people) vs the tuning set."""
    labels = [f"{t:g}" for t in full_sample["p_threshold"]]
    x_positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x_positions, full_sample["r_squared"], marker="o", color="lightsteelblue", linewidth=2,
            label="All people (fit and scored on the same people)")
    ax.plot(x_positions, tuning["r_squared"], marker="o", color="steelblue", linewidth=2,
            label="Tuning set (held out)")
    ax.set_xticks(x_positions, labels)
    ax.set_xlabel("p-value threshold")
    ax.set_ylabel("R² (predicted vs true expression)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{gene_label} expression score by p-value threshold", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
