# Results summary

This document summarizes the current finalized results for the TCGA-UCEC DNA methylation biomarker project.

## Dataset overview

| Dataset | Platform | Tumor | Normal | Notes |
|---|---:|---:|---:|---|
| TCGA-UCEC | Illumina HumanMethylation450 | 420 | 46 | Discovery and internal validation |
| GSE155760 | Illumina HumanMethylationEPIC | 33 | 13 | External validation; endometrial carcinoma vs cancer-free normal endometrial mucosa |

Cleaned TCGA matrix:

- 390,516 probes × 466 samples
- Stored outside this GitHub-ready repository as `.npy`
- Not copied into the repository because it is large and should be regenerated/downloaded by scripts

## Main panel

Final recommended panel: literature-constrained + data-driven 10-gene methylation panel.

| Gene | Probe | Source | TCGA Δβ |
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

All selected markers are promoter-region hypermethylated in tumors.

## Internal validation

Fully nested TCGA evaluation:

- Outer 5-fold stratified CV
- Differential methylation analysis recomputed inside each outer training fold
- Promoter hypermethylated DMP filtering recomputed inside each outer training fold
- LassoCV feature selection recomputed inside each outer training fold
- Logistic regression evaluated on held-out validation fold

| Metric | Value |
|---|---:|
| AUC | 0.9989 ± 0.0024 |
| Sensitivity | 0.9976 |
| Specificity | 0.9778 |
| Accuracy | 0.9957 |

This result suggests the high internal performance is not mainly caused by feature-selection leakage.

## Fixed-panel comparison

Same 5-fold CV workflow for three fixed panels:

| Panel | Features | AUC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| Literature-constrained + data-driven 10-gene panel | 10 | 0.9992 ± 0.0018 | 0.9976 | 0.9556 |
| Pure data-driven 10-gene panel | 10 | 0.9989 ± 0.0024 | 0.9976 | 0.9778 |
| PAX1/JAM3 baseline | 2 | 0.7225 ± 0.0251 | 1.0000 | 0.0000 |

The literature-constrained panel preserves near-identical internal performance while improving biological interpretability.

## External validation

External validation was performed on GEO GSE155760.

Validation subset:

- Tumor: 33
  - Endometrial endometrioid carcinoma: 23
  - Uterine serous carcinoma: 10
- Normal: 13
  - Cancer-free normal endometrial mucosa: 13

All 10 panel probes were present on the EPIC array.

| Dataset | AUC | Sensitivity | Specificity | TN/FP/FN/TP |
|---|---:|---:|---:|---|
| GSE155760 | 0.9790 | 0.9091 | 1.0000 | 13/0/3/30 |

The external AUC decreased from the TCGA internal estimate, as expected, but remained high. Notably, the 13 normal endometrial samples had zero false positives at the default 0.5 threshold.

## New data-driven marker strength

Four data-driven markers were evaluated individually in TCGA and GSE155760.

| Gene | Probe | TCGA AUC | GEO AUC | Interpretation |
|---|---|---:|---:|---|
| CYP26C1 | cg05219493 | 0.997 | 0.991 | Very strong |
| SPARCL1 | cg08003102 | 0.986 | 0.981 | Very strong |
| WDR52 | cg24199400 | 0.983 | 0.946 | Strong |
| CLDN15 | cg24809529 | 0.978 | 0.925 | Moderate-to-strong |

## Batch-effect sensitivity analysis

PCA revealed some plate/TSS-associated structure and partial batch-group confounding. ComBat reduced plate-associated PCA structure, but biological tumor-normal separation remained strong.

Interpretation:

- Batch effects exist and should be reported.
- The tumor-normal methylation signal is not fully explained by batch.
- External validation remains essential for translational claims.

## Recommended manuscript framing

Best current framing:

> A literature-constrained and data-driven DNA methylation panel for endometrial cancer detection, developed from TCGA-UCEC and externally validated in GSE155760.

Avoid overclaiming:

- Current data are tissue-based public datasets.
- The model is not yet clinically validated for cervical scrapings, endometrial brushings, plasma cfDNA, or screening populations.

Recommended next validation:

- qMSP, pyrosequencing, or targeted bisulfite sequencing in independent tissue samples.
- Ideally, validation in cervical or endometrial brushing samples.
- Head-to-head comparison against CDO1/CELF4 and PAX1/JAM3 assays.
