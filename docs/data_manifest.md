# Data manifest

This file records where the project inputs and derived outputs are stored locally, and which files should or should not be committed to GitHub.

## Local source analysis directory

Primary working directory:

```text
/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation
```

GitHub-ready project directory:

```text
/Users/wuwenbo/AsiaInfo/UCEC_methylation_biomarker
```

## Large files not copied into the GitHub-ready repository

These files are required for rerunning the full analysis but should not be committed to GitHub.

| File | Description | Approximate status |
|---|---|---|
| `/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation/tcga_ucec_hm450_cleaned/matrix/ucec_hm450_beta_clean_probes_by_samples.npy` | Cleaned TCGA beta matrix, probes × samples | Large binary matrix |
| `/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation/tcga_ucec_hm450_gdc/matrix/ucec_hm450_beta.npy` | Raw assembled TCGA beta matrix | Large binary matrix |
| `/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation/tcga_ucec_hm450_gdc/raw_gdc_files/` | Raw GDC methylation files | Large directory |
| `/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation/geo_external_validation/GSE155760/raw/GSE155760_signals.csv.gz` | GEO GSE155760 processed EPIC signal matrix | ~491 MB |
| `/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation/tcga_ucec_hm450_batch_effect/matrix/ucec_hm450_beta_combat_plate_probes_by_samples.npy` | ComBat-corrected beta matrix | Large binary matrix |

Recommended policy:

- Do not upload `.npy` matrices or raw GEO/GDC files to GitHub.
- Keep download scripts, configuration files, small summary tables, and figures in the repository.
- For public sharing, provide instructions to regenerate large files from GDC/GEO.

## Key small outputs copied into this repository

### Core results

```text
outputs/tables/core/final_literature_constrained_10gene_panel.parquet
outputs/tables/core/new_4_data_driven_marker_quantitative_strength.parquet
outputs/tables/core/final_10_marker_summary_original_data_driven.parquet
outputs/tables/core/key_results_summary_original_panel.parquet
```

### Validation results

```text
outputs/tables/validation/fully_nested_outer_fold_metrics.parquet
outputs/tables/validation/fully_nested_training_fold_dmp_counts.parquet
outputs/tables/validation/fully_nested_selected_features_by_fold.parquet
outputs/tables/validation/three_panel_5fold_cv_summary.parquet
outputs/tables/validation/three_panel_5fold_cv_fold_metrics.parquet
outputs/tables/validation/GSE155760_external_validation_metrics.parquet
outputs/tables/validation/GSE155760_external_validation_predictions.parquet
outputs/tables/validation/GSE155760_missing_panel_probes.parquet
outputs/tables/validation/GSE155760_endometrial_validation_samples.parquet
outputs/tables/validation/delong_auc_test_10marker_vs_pax1_jam3.parquet
outputs/tables/validation/lr_10marker_vs_pax1_jam3_test_metrics.parquet
```

### Annotation and candidate results

```text
outputs/tables/annotation/ucec_hm450_differential_methylation_significant.parquet
outputs/tables/annotation/ucec_hm450_dmp_promoter_TSS200_TSS1500.parquet
outputs/tables/annotation/ucec_promoter_hypermethylated_candidate_genes_dedup.parquet
outputs/tables/annotation/known_marker_overlaps_with_candidate_genes.parquet
```

### QC results

```text
outputs/tables/qc/pca_batch_association_before_combat.parquet
outputs/tables/qc/pca_batch_association_after_combat.parquet
```

### Reports

```text
outputs/reports/fully_nested_cv_report.json
outputs/reports/literature_constrained_panel_report.json
outputs/reports/GSE155760_external_validation_report.json
outputs/reports/batch_effect_report.json
```

### Model

```text
models/tcga_trained_lr_pipeline_literature_constrained_10gene.pkl
```

This model was trained on TCGA using the fixed literature-constrained 10-gene panel and used for GSE155760 external validation when no separate model parameter file was supplied.

## Backup recommendation

Recommended local archive name:

```text
UCEC_methylation_biomarker_archive_2026-09-06
```

Suggested backups:

1. GitHub repository for code, documents, figures, and small result tables.
2. Local compressed archive for the complete project, including large `.npy` matrices.
3. External drive or institutional storage for raw GDC/GEO downloads.
4. Cloud storage for manuscript-ready figures and final tables.

For GitHub, use `.gitignore` to exclude:

```text
*.npy
*.csv.gz
*.idat
data/raw/
data/processed/
geo_external_validation/*/raw/
.conda-env/
.miniforge3/
```
