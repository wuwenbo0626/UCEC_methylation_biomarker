# Manuscript progress log

Date: 2026-09-07  
Project: UCEC methylation biomarker panel  
Repository: <https://github.com/wuwenbo0626/UCEC_methylation_biomarker>

## Current manuscript status

The project has entered the manuscript-preparation stage. Core computational analyses have been completed, the reproducible GitHub repository has been organized, and an initial preprint-style manuscript draft has been created.

Current manuscript level:

- Not yet submission-ready.
- Suitable as a first complete draft for advisor/collaborator review.
- Needs reference verification, figure/table alignment, and language polishing before medRxiv or journal submission.

## Manuscript files created

The manuscript workspace is located at:

```text
manuscript/
```

Current files:

```text
manuscript/README.md
manuscript/manuscript_draft.md
manuscript/preprint_v1.md
manuscript/references.bib
manuscript/masters_application_abstract.md
```

File purposes:

| File | Purpose | Status |
|---|---|---|
| `manuscript_draft.md` | First full medRxiv-style manuscript draft | Completed as rough draft |
| `preprint_v1.md` | Cleaner and more formal preprint v1 | Completed as current main draft |
| `references.bib` | BibTeX references for manuscript | Core references added; needs final manual check |
| `masters_application_abstract.md` | English research summary for graduate application | Completed initial version |
| `README.md` | Notes for manuscript workspace | Updated |

## Current manuscript title

```text
A literature-constrained and data-driven DNA methylation panel for endometrial cancer detection
```

Current short title:

```text
DNA methylation panel for endometrial cancer detection
```

## Core study framing

Recommended framing:

> This is a computational biomarker discovery and external validation study using public tissue-based methylation-array datasets.

Recommended wording:

- Use “detection” rather than “screening” in the title and main claims.
- Say the panel is promising for future assay development.
- Avoid claiming clinical screening readiness.
- Emphasize that future validation is needed in targeted assays and minimally invasive clinical samples.

## Dataset summary

| Dataset | Platform | Tumor | Normal | Role |
|---|---|---:|---:|---|
| TCGA-UCEC | Illumina HumanMethylation450 | 420 | 46 | Discovery and internal validation |
| GSE155760 | Illumina EPIC | 33 | 13 | External validation |

Cleaned TCGA matrix:

- 390,516 probes
- 466 samples

## Differential methylation results

Current numbers included in `preprint_v1.md`:

| Item | Count |
|---|---:|
| Significant differentially methylated probes | 50,919 |
| Tumor-hypermethylated DMPs | 21,414 |
| Tumor-hypomethylated DMPs | 29,505 |
| Promoter DMPs in TSS200/TSS1500 | 10,207 |
| Promoter hypermethylated probes | 4,848 |
| Deduplicated promoter hypermethylated candidate genes | 1,697 |

Important interpretation:

- Genome-wide significant DMPs include more hypomethylated than hypermethylated probes.
- The panel-building strategy intentionally focuses on promoter hypermethylation because it is biologically interpretable and suitable for targeted methylation assays.

## Final 10-gene panel

| Gene | Probe | Source | TCGA delta beta |
|---|---|---|---:|
| CDO1 | cg23180938 | Literature-constrained | 0.5970 |
| PAX1 | cg17620199 | Literature-constrained | 0.3136 |
| BHLHE22 | cg06873806 | Literature-constrained | 0.3413 |
| HAND2 | cg19178853 | Literature-constrained | 0.4158 |
| TBX5 | cg06911121 | Literature-constrained | 0.4793 |
| ZNF454 | cg24843380 | Literature-constrained | 0.6393 |
| CYP26C1 | cg05219493 | Data-driven | 0.3935 |
| SPARCL1 | cg08003102 | Data-driven | 0.5185 |
| WDR52 | cg24199400 | Data-driven | 0.4138 |
| CLDN15 | cg24809529 | Data-driven | 0.3772 |

Current interpretation:

- Six genes are literature-constrained.
- Four genes are data-driven additions.
- All ten selected markers are promoter-region hypermethylated in TCGA-UCEC tumors.

## Model performance included in manuscript

| Evaluation | AUC | Sensitivity | Specificity | Notes |
|---|---:|---:|---:|---|
| Fully nested TCGA 5-fold CV | 0.9989 ± 0.0024 | 0.9976 | 0.9778 | Feature discovery repeated inside training folds |
| Fixed 10-gene TCGA 5-fold CV | 0.9992 ± 0.0018 | 0.9976 | 0.9556 | Literature-constrained plus data-driven panel |
| Pure data-driven TCGA 5-fold CV | 0.9989 ± 0.0024 | 0.9976 | 0.9778 | Exploratory comparator |
| PAX1/JAM3 TCGA baseline | 0.7225 ± 0.0251 | 1.0000 | 0.0000 | Baseline under evaluated threshold |
| GSE155760 external validation | 0.9790 | 0.9091 | 1.0000 | 33 tumor and 13 normal samples |

DeLong comparison:

- 10-marker model AUC: 0.9966
- PAX1/JAM3 baseline AUC: 0.7115
- AUC difference: 0.2851
- z = 6.2635
- p = 3.76 × 10^-10

## Figures currently referenced

The manuscript currently references:

| Figure | Content | Existing repository file |
|---|---|---|
| Figure 1 | Study workflow | `outputs/figures/main/fig1_study_workflow.*` |
| Figure 2 | Volcano plot | `outputs/figures/main/fig2_volcano_plot.*` |
| Figure 3 | Marker beta-value boxplots | `outputs/figures/main/fig3_marker_beta_boxplots.*` |
| Figure 4 | ROC comparison | `outputs/figures/main/fig4_roc_10marker_vs_baseline.*` |
| Figure 5 | Logistic regression coefficients | `outputs/figures/main/fig5_lr_feature_coefficients.*` |
| Figure 6 | Confusion matrices | `outputs/figures/main/fig6_model_confusion_matrices.*` |

All main figures currently exist as PNG and PDF.

## References status

Core references added to `manuscript/references.bib` include:

- TCGA-UCEC integrated genomic characterization.
- Genomic Data Commons.
- TCGAbiolinks.
- GSE155760/GEO.
- Methylomic analysis linked to GSE155760.
- HAND2 methylation in endometrial cancer.
- MPap/cervical DNA methylation assay for endometrial cancer detection.
- Benjamini-Hochberg FDR.
- LASSO.
- DeLong AUC comparison.
- ComBat and sva.
- scikit-learn, numpy, pandas, scipy.

Reference caution:

- Marker-specific references still need one final manual verification before submission.
- Do not submit until all author lists, titles, journal names, years, volumes, pages, PMIDs, and DOIs are checked.

## Current strengths of the manuscript

1. Clear public-data discovery and validation structure.
2. Strong internal performance with leakage-aware nested cross-validation.
3. External validation in GSE155760.
4. Biologically interpretable hybrid panel: literature-constrained + data-driven.
5. Reproducible GitHub repository with scripts, config, selected outputs, manuscript files, and citation metadata.

## Current limitations to state clearly

1. Current evidence is computational and tissue-based.
2. The study does not prove clinical screening performance.
3. Normal sample size is limited, especially in GSE155760.
4. Batch effects exist and cannot be fully excluded.
5. Array probes need targeted assay validation.
6. Default logistic-regression threshold is not a clinically optimized cutoff.

## Recommended next steps

### Step 1: Reference audit

Manually verify every citation in `references.bib`, especially marker-specific studies.

### Step 2: Figure/table alignment

Check that each figure and table mentioned in `preprint_v1.md` exactly matches the final output files.

### Step 3: Manuscript polishing

Convert the manuscript from Markdown to Word/PDF and polish:

- abstract length;
- Methods clarity;
- Results order;
- Discussion tone;
- limitation wording;
- journal/preprint formatting.

### Step 4: Optional supplementary materials

Create supplementary tables:

- full 10-gene marker table;
- nested CV fold-level results;
- GEO validation predictions;
- batch-effect sensitivity analysis;
- selected feature stability across folds.

### Step 5: Experimental validation plan

Draft a short wet-lab validation proposal:

- targeted bisulfite sequencing or qMSP;
- independent endometrial tissue samples;
- cervical/endometrial brushing samples if available;
- head-to-head comparison with CDO1/CELF4 and PAX1/JAM3 baselines.

## Git status at time of this log

Latest known commits before this progress log:

```text
13a0175 Add preprint v1 manuscript
4f8ebc7 Add manuscript draft and application abstract
6a14522 Add repository publication metadata and reproducibility checklist
00678d5 Initial reproducible UCEC methylation biomarker project
```

This progress log was created to make the manuscript state easy to recover in future sessions.

