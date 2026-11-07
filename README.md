# Genetic Risk Prediction: Quarter 1 Project

DSC 180A, Fall 2026. Domain: genetic risk prediction of gene expression, complex traits, and polygenic disease (mentor: Tiffany Amariuta).

This project asks how well a person's DNA predicts their traits. It maps cis-eQTLs (SNPs linked to how much a gene is expressed) in 1000 Genomes European samples, builds clumping and thresholding (C+T) scores that predict gene expression from genotype, and builds a height polygenic risk score (PRS) for the 1000 Genomes panel and one 23andMe sample.

The `notebooks/` folder has one notebook per week that walks through the same analyses step by step.

## Data

The raw data is not in this repo (`data/` is in `.gitignore`). Put the files in `data/raw/`:

1. **Genotypes:** 1000 Genomes European samples in PLINK format. Download [LDREF.tar.bz2](https://data.broadinstitute.org/alkesgroup/FUSION/LDREF.tar.bz2) and unzip it so the files sit at `data/raw/LDREF/1000G.EUR.<chromosome>.bed/.bim/.fam`.
2. **Gene expression:** Geuvadis LCL RNA-seq, `GD462.GeneQuantRPKM.50FN.samplename.resk10.txt` (provided by the mentor), saved in `data/raw/`.
3. **Height GWAS summary statistics:** `Height.gwas.txt.gz` from the [PRS tutorial](https://choishingwan.github.io/PRS-Tutorial/), saved in `data/raw/`.
4. **23andMe file (optional):** saved as `data/raw/tiffany_23andme.txt`. This is personal genetic data shared by the mentor for the course, so it is not public. Without it, the `prs` target scores only the 1000 Genomes panel.

File paths can be changed in `config/data-params.json` and `config/prs-params.json` without touching the code.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Running it

```bash
python run.py test      # whole pipeline on small made-up data in test/testdata (no real data needed)
python test/check_test_outputs.py   # checks the test run got the hand-worked answers
python run.py all       # whole pipeline on the real data (a few minutes on a laptop)
python run.py data      # Weeks 1-2: file descriptions and sample overlap
python run.py explore   # Weeks 2-3: MAF groups and SNPs per cis window
python run.py eqtl      # Week 3: cis-eQTL scan for one gene
python run.py gene-prs  # Week 4: chromosome-wide eQTLs and cross-validated expression scores
python run.py prs       # Week 5: height PRS
python run.py clean     # delete outputs/ and test/outputs/
```

Results go to `outputs/` (real data) or `test/outputs/` (test data), along with a `run.log`.

To scan a different gene, change `gene_id` and `gene_label` in `config/eqtl-params.json`. To scan a different chromosome in Week 4, change `chromosome` in `config/gene-prs-params.json`.

## Methods

- **Sample matching:** keep people whose ID appears in both the genotype `.fam` file and the expression file columns (344 people), in genotype file order.
- **Genotype coding:** copies (0, 1, 2) of the `.bim` file's first allele. Missing calls are filled with that SNP's average.
- **MAF groups:** rare MAF < 1%, low frequency 1% to 5%, common > 5%.
- **cis window:** SNPs within 500 kb of the gene position in the expression file (`Coord`).
- **eQTL test:** simple linear regression, expression ~ copies, one SNP at a time. Significance uses a Bonferroni cutoff of 0.05 / (number of SNPs tested for that gene).
- **C+T scores:** clump SNPs (LD r² > 0.1 within 250 kb, LD from the genotype panel; for Week 4, from the training people only), keep clumped SNPs below the p-value threshold, score = sum of effect size x copies.
- **Week 4 cross-validation:** one random split (seed in config) of the 344 people into 275 training, 34 tuning, and 35 testing. Effect sizes come from training, the threshold with the best tuning R² is chosen, and R² is reported on the testing people.
- **Week 5 GWAS QC:** INFO > 0.8, MAF > 0.01, drop strand-ambiguous (A/T, C/G) and duplicate SNPs, beta = log(OR). GWAS alleles are aligned to the 1000 Genomes counted allele (flipping the sign when swapped, and checking the opposite strand).
- **Week 5 with a 23andMe sample:** the score is rebuilt using only SNPs the person was genotyped at, and the 1000 Genomes panel is rescored on that same set, so everyone is compared on the same SNPs.

## Limitations

- The cis window is centered on the single `Coord` value in the expression file, not the full gene body. `gene_annot.txt.gz` would fix this.
- LDREF is about 1.2 million mostly common SNPs (HapMap3), not all 1000 Genomes variants, so rare variants are mostly missing.
- No covariates (sex, ancestry PCs) in the regression. The expression file already has 10 hidden factors removed ("resk10").
- Week 4's testing group is 35 people, so each gene's test accuracy is noisy. The distribution across genes is more reliable than any single gene.
- The height PRS has no sex term and the panel mixes men and women, so a percentile ranks genetic score, not height within one sex.

## AI Usage and Contributions

I used AI tools, primarily ChatGPT, to help organize week-to-week notebook and progress-report templates, explain background biology and genetics concepts, and improve the clarity of written documentation. I was responsible for making research and methodological decisions, implementing and testing the analysis pipeline, running the analyses, interpreting the results, and reviewing the final work for accuracy.
