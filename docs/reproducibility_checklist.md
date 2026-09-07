# Reproducibility checklist

This checklist records what is currently reproducible in this repository and what depends on locally stored large data files.

## Repository status

- [x] Code is organized into numbered scripts.
- [x] Main configuration is stored in `config.yaml`.
- [x] Python dependencies are listed in `requirements.txt`.
- [x] Key lightweight results are stored as `.parquet` or `.json`.
- [x] Main figures are stored as both `.png` and `.pdf`.
- [x] Large matrices and raw downloads are excluded by `.gitignore`.
- [x] Local data locations are documented in `docs/data_manifest.md`.
- [x] GitHub repository is public.

## Environment

Recommended environment:

```bash
conda create -n ucec_methylation python=3.11 -c conda-forge -y
conda activate ucec_methylation
python -m pip install -r requirements.txt
```

The workflow was developed on:

- macOS on Apple Silicon
- MacBook Pro M2
- 16 GB RAM
- Python 3.11

## Data reproducibility

The full analysis requires large public datasets that are not committed to GitHub:

- TCGA-UCEC HumanMethylation450 beta values from GDC
- GEO GSE155760 EPIC processed signal matrix
- Illumina HumanMethylation450 manifest

Expected local paths are documented in `docs/data_manifest.md`.

## Analysis-level reproducibility

Core analysis modules:

| Step | Script |
|---|---|
| Download TCGA-UCEC methylation data | `scripts/01_download_data.py` |
| Preprocess beta matrix | `scripts/02_preprocess.py` |
| Differential methylation analysis | `scripts/03_differential_analysis.py` |
| Probe annotation | `scripts/04_annotate_probes.py` |
| Enrichment analysis | `scripts/05_enrichment_analysis.py` |
| Feature selection | `scripts/06_feature_selection.py` |
| Model training | `scripts/07_train_model.py` |
| Baseline comparison/report generation | `scripts/08_evaluate_model.py` |
| Batch-effect sensitivity analysis | `scripts/09_assess_batch_effects_combat.py` |
| Nested LASSO/logistic regression validation | `scripts/10_nested_cv_lasso_lr_promoter_hyper.py` |
| Literature-constrained panel construction | `scripts/11_literature_constrained_10gene_panel.py` |
| Fully nested DMP + LASSO + LR validation | `scripts/12_fully_nested_cv_dmp_lasso_lr.py` |
| GEO GSE155760 external validation | `scripts/13_external_validate_geo_gse155760.py` |

## Randomness control

- `random_state=42` is used for cross-validation splits and model training where applicable.
- Random forest and cross-validation jobs use `n_jobs=4` where relevant.
- Cross-validation outputs store scalar metrics and selected feature lists, not full prediction vectors.

## File format policy

- Large matrices: `.npy`
- Result tables: `.parquet`
- Figures: 300 dpi `.png` plus `.pdf`
- Reports: `.json` and `.md`
- Large CSV exports are avoided.

## Known limitations

- TCGA and GSE155760 are tissue-based datasets, not true screening specimens.
- External validation sample size is modest, especially for normal controls.
- The panel is not clinically validated for cervical scrapings, endometrial brushings, lavage, or cfDNA.
- Batch effects exist in TCGA and should be reported in manuscripts.

## Recommended next checks before manuscript submission

- [ ] Re-run the repository from a clean environment.
- [ ] Confirm all figure labels are publication-ready.
- [ ] Add a manuscript draft under `manuscript/`.
- [ ] Add exact software versions from `pip freeze` or `conda env export`.
- [ ] Consider archiving a release on Zenodo after the manuscript is ready.
