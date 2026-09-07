# A literature-constrained and data-driven DNA methylation panel for endometrial cancer detection

**Short title:** DNA methylation panel for endometrial cancer detection  
**Author:** Wuwenbo Wu  
**Correspondence:** Wuwenbo Wu  
**Draft version:** Preprint v1, 2026-09-07  
**Repository:** <https://github.com/wuwenbo0626/UCEC_methylation_biomarker>

## Abstract

### Background

DNA methylation is a promising biomarker modality for endometrial cancer because tumor-associated promoter hypermethylation can be stable, biologically interpretable, and measurable using targeted assays. However, high-dimensional methylation-array studies are vulnerable to overfitting, particularly when feature selection is performed before model validation. We aimed to develop an interpretable methylation marker panel for endometrial cancer detection using a workflow that combines prior literature constraints, data-driven feature selection, leakage-aware internal validation, and external validation.

### Methods

We analyzed TCGA-UCEC Illumina HumanMethylation450 data from 420 primary tumor and 46 solid tissue normal samples. After quality control, 390,516 probes across 466 samples were retained. Differential methylation analysis compared tumor and normal samples using beta-to-M-value transformation, two-sample t-tests, Benjamini-Hochberg false discovery rate correction, and delta-beta estimation. Candidate markers were restricted to promoter-region hypermethylated probes in TSS200 or TSS1500 regions. A 10-gene panel was constructed by forcing inclusion of six literature-supported genes and using LASSO to select four additional data-driven genes. Logistic regression was used as the final classifier. Internal performance was evaluated using fully nested 5-fold cross-validation, in which differential methylation filtering, promoter restriction, LASSO feature selection, and model training were repeated inside each outer training fold. External validation was performed in a curated subset of GEO GSE155760 containing 33 endometrial cancer and 13 normal endometrial samples.

### Results

The final panel included six literature-constrained genes (*CDO1*, *PAX1*, *BHLHE22*, *HAND2*, *TBX5*, and *ZNF454*) and four data-driven genes (*CYP26C1*, *SPARCL1*, *WDR52*, and *CLDN15*). All representative probes were promoter-associated and hypermethylated in TCGA-UCEC tumors. Fully nested TCGA cross-validation achieved an AUC of 0.9989 ± 0.0024, sensitivity of 0.9976, and specificity of 0.9778. The fixed 10-gene panel achieved a 5-fold TCGA AUC of 0.9992 ± 0.0018, sensitivity of 0.9976, and specificity of 0.9556. External validation in GSE155760 achieved an AUC of 0.9790, sensitivity of 0.9091, and specificity of 1.0000. Compared with a *PAX1*/*JAM3* two-gene baseline, the 10-marker model showed significantly higher AUC in internal comparison (DeLong p = 3.76 × 10^-10).

### Conclusions

We developed a biologically interpretable 10-gene promoter methylation panel for endometrial cancer detection and observed strong internal and external performance in public tissue-based methylation-array cohorts. These findings support further validation using targeted methylation assays in independent clinical specimens, especially minimally invasive sample types such as endometrial or cervical brushing. The current study should be interpreted as computational biomarker discovery and validation, not as evidence of clinical screening readiness.

## Keywords

Endometrial cancer; DNA methylation; promoter hypermethylation; TCGA-UCEC; biomarker discovery; LASSO; logistic regression; external validation.

## Introduction

Endometrial cancer is a common gynecologic malignancy with rising incidence in many populations. Molecular characterization has already reshaped understanding of the disease, including through the TCGA-UCEC study, which established major genomic and molecular subgroups of endometrial carcinoma [@tcga2013ucec]. Alongside molecular classification, there is growing interest in biomarkers that could support early detection, diagnostic triage, and risk-adapted clinical decision-making.

DNA methylation is a strong candidate technology for such biomarker development. Aberrant DNA methylation is widespread in cancer, and promoter hypermethylation can reflect regulatory disruption at cancer-relevant genes. From a translational perspective, methylation markers also have practical advantages: they can be measured with quantitative methylation-specific PCR, pyrosequencing, targeted bisulfite sequencing, or related assays, and may be compatible with low-input or minimally invasive clinical samples.

Several methylation markers have already been reported in endometrial cancer detection studies. For example, *HAND2* methylation has been implicated in endometrial carcinogenesis [@jones2013hand2], and panels involving genes such as *BHLHE22*, *CDO1*, *CELF4*, and related loci have been evaluated in cervical scrapings or abnormal uterine bleeding cohorts [@huang2017cervicalscrapings; @wen2022mpap]. These studies suggest that methylation-based detection of endometrial cancer is biologically and clinically plausible. Nevertheless, translating genome-wide methylation discovery into a robust and interpretable marker panel remains difficult.

One problem is statistical overfitting. Methylation arrays measure hundreds of thousands of CpG probes, while normal endometrial controls in public cancer datasets are often limited. If feature selection is performed on the full dataset before cross-validation or train-test splitting, model performance can be inflated by information leakage. Another problem is interpretability. A purely data-driven model may select probes with excellent classification performance but unclear biological or translational relevance, making follow-up assay development less persuasive.

We therefore designed a hybrid strategy: constrain part of the model using prior marker evidence, then complete the panel with data-driven selection from promoter-region hypermethylated probes. The resulting model was evaluated using a fully nested cross-validation design and an external GEO validation cohort. The aim was to produce a reproducible and experimentally actionable candidate panel for future validation, while avoiding overclaiming beyond the limits of public tissue-based datasets.

## Methods

### Study design and data sources

This was a retrospective computational biomarker discovery and validation study using publicly available DNA methylation-array datasets. TCGA-UCEC was used for discovery and internal validation. TCGA data were accessed through the Genomic Data Commons, a shared platform for cancer genomic data [@grossman2016gdc]. GEO GSE155760 was used for external validation [@geo_gse155760]. GSE155760 was originally generated in a methylomic study of gynecologic tissues and ovarian cancer-related lesions [@pisanic2018methylomic]; for this project, we curated the subset corresponding to endometrial carcinoma and normal endometrial mucosa.

### TCGA-UCEC methylation data

The TCGA-UCEC methylation dataset was profiled on the Illumina HumanMethylation450 array. The final included samples were 420 primary tumor samples and 46 solid tissue normal samples. After preprocessing, the cleaned beta-value matrix contained 390,516 probes and 466 samples.

### External validation data

The external validation subset from GSE155760 included 33 endometrial cancer samples and 13 normal endometrial mucosa samples. Samples were profiled on the Illumina EPIC methylation array. All ten final panel probes were present in the external validation matrix.

### Preprocessing and quality control

Methylation beta values were represented as probe-by-sample matrices. TCGA sample groups were assigned using barcode-derived sample type information. Probes with more than 10% missing values and samples with more than 5% missing values were removed. SNP probes beginning with `rs` and probes mapping to sex chromosomes were excluded. Remaining missing values were imputed using k-nearest-neighbor imputation with k = 5. Large matrices were stored as `.npy` arrays, and metadata tables were stored as `.parquet` files.

### Differential methylation analysis

For statistical testing, beta values were clipped to the interval [1e-6, 0.999999] and transformed to M values:

```text
M = log2(beta / (1 - beta)).
```

For each probe, tumor and normal groups were compared using two-sample t-tests on M values. Delta beta was calculated as mean beta in tumors minus mean beta in normals. P values were adjusted using the Benjamini-Hochberg procedure [@benjamini1995fdr]. Genome-wide significant differentially methylated probes were defined as probes with |delta beta| > 0.2 and FDR < 0.05.

### Probe annotation and candidate filtering

HumanMethylation450 probes were annotated using Illumina manifest-derived probe annotation. Promoter probes were defined as probes annotated to TSS200 or TSS1500. The candidate biomarker pool was restricted to tumor-hypermethylated promoter probes, defined as delta beta > 0.2 and FDR < 0.05. For final literature-constrained panel construction, we further required selected genes to show tumor hypermethylation with delta beta > 0.3.

### Literature-constrained and data-driven panel selection

The final panel was constructed in two stages. First, we forced inclusion of six literature-supported or biologically plausible genes: *CDO1*, *PAX1*, *BHLHE22*, *HAND2*, *TBX5*, and *ZNF454*. For each gene, a representative promoter probe was selected from available tumor-hypermethylated candidates, prioritizing variance across samples when multiple probes were available. Second, LASSO feature selection was applied to the remaining candidate pool to select four additional data-driven markers. LASSO was implemented using scikit-learn and is based on L1-penalized regression [@tibshirani1996lasso; @pedregosa2011sklearn]. Pseudogenes, genes with unclear functional annotation, and previously selected biologically ambiguous candidates were excluded from this stage.

### Model development

The final classifier was logistic regression using beta values from the ten selected probes. Logistic regression was prioritized because it is interpretable, simple to recalibrate, and more directly translatable to a targeted assay score than more complex models. During exploratory modeling, logistic regression, random forest, XGBoost, and support-vector-machine classifiers were compared, but logistic regression was retained because its discrimination was not materially inferior and it offered clearer interpretability.

### Leakage-aware internal validation

To estimate internal performance without feature-selection leakage, we implemented fully nested 5-fold stratified cross-validation. In each outer training fold, differential methylation testing, FDR correction, promoter hypermethylation filtering, LASSO feature selection, and logistic regression training were recomputed using only training samples. The trained model was then applied to the held-out outer validation fold. Metrics included AUC, sensitivity, specificity, accuracy, F1 score, and confusion matrix counts. AUC comparison between the 10-marker model and the *PAX1*/*JAM3* baseline used DeLong's test [@delong1988roc].

### Fixed-panel and baseline comparison

The final literature-constrained plus data-driven 10-gene panel was compared with two reference panels: a pure data-driven 10-gene panel and a *PAX1*/*JAM3* two-gene baseline. Each panel was evaluated using 5-fold stratified cross-validation with logistic regression. These comparisons were used to assess the tradeoff between predictive performance, interpretability, and prior biological support.

### Batch-effect sensitivity analysis

Available metadata were inspected for potential technical variables, including plate and related batch-like fields. Principal component analysis was used to visualize methylation structure by biological group and batch variable. ComBat was explored as a sensitivity analysis for batch correction [@johnson2007combat; @leek2012sva]. The batch analysis was interpreted descriptively because tumor-normal group membership can be partially confounded with technical variables in public cohorts.

### Software and reproducibility

Analyses were implemented in Python 3.11 using numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib, seaborn, pyarrow, joblib, xgboost, and gseapy [@harris2020numpy; @mckinney2010pandas; @virtanen2020scipy; @pedregosa2011sklearn]. Scripts, configuration files, processed result tables, selected figures, and model artifacts are organized in a public GitHub repository. Large raw and processed methylation matrices are excluded from the repository and can be regenerated using the provided scripts.

## Results

### TCGA-UCEC methylation dataset and preprocessing

The final TCGA-UCEC analysis included 466 samples: 420 primary tumor and 46 solid tissue normal samples. After quality control, filtering, and imputation, 390,516 probes remained. This cleaned matrix was used for downstream differential methylation analysis and model development.

### Differential methylation landscape

Genome-wide differential methylation analysis identified 50,919 significant differentially methylated probes using |delta beta| > 0.2 and FDR < 0.05. Of these, 21,414 were hypermethylated in tumors and 29,505 were hypomethylated in tumors. Although hypomethylated probes were more numerous genome-wide, the biomarker discovery strategy intentionally focused on promoter-region hypermethylation because promoter hypermethylation is assay-friendly and biologically interpretable for cancer detection. Restricting to TSS200/TSS1500 promoter probes identified 10,207 promoter differentially methylated probes, including 4,848 promoter hypermethylated probes. After gene-level de-duplication, 1,697 promoter-hypermethylated candidate genes remained.

### Final 10-gene panel

The final panel contained six literature-constrained markers and four data-driven markers. All ten markers were promoter-region hypermethylated in TCGA-UCEC tumors, with delta beta values ranging from 0.3136 to 0.6393.

| Rank | Gene | Probe | Source | Promoter relation | TCGA delta beta |
|---:|---|---|---|---|---:|
| 1 | *CDO1* | cg23180938 | Literature-constrained | TSS200 | 0.5970 |
| 2 | *PAX1* | cg17620199 | Literature-constrained | TSS1500 | 0.3136 |
| 3 | *BHLHE22* | cg06873806 | Literature-constrained | TSS200 | 0.3413 |
| 4 | *HAND2* | cg19178853 | Literature-constrained | TSS1500/1stExon | 0.4158 |
| 5 | *TBX5* | cg06911121 | Literature-constrained | TSS1500 | 0.4793 |
| 6 | *ZNF454* | cg24843380 | Literature-constrained | TSS1500 | 0.6393 |
| 7 | *CYP26C1* | cg05219493 | Data-driven | TSS200 | 0.3935 |
| 8 | *SPARCL1* | cg08003102 | Data-driven | TSS1500 | 0.5185 |
| 9 | *WDR52* | cg24199400 | Data-driven | TSS200 | 0.4138 |
| 10 | *CLDN15* | cg24809529 | Data-driven | TSS200 | 0.3772 |

### Fully nested internal validation

Fully nested 5-fold cross-validation was performed to assess model performance without feature-selection leakage. In each outer fold, the entire feature-discovery process was repeated using only the training set. The model achieved a mean AUC of 0.9989 ± 0.0024, mean sensitivity of 0.9976, mean specificity of 0.9778, and mean accuracy of 0.9957. These results indicate that the high internal discrimination was not primarily caused by selecting features on the full dataset before validation.

### Fixed-panel comparison

The fixed literature-constrained plus data-driven 10-gene panel achieved an AUC of 0.9992 ± 0.0018, sensitivity of 0.9976, specificity of 0.9556, and accuracy of 0.9935 in TCGA 5-fold cross-validation. A pure data-driven 10-gene panel produced similar discrimination, with AUC 0.9989 ± 0.0024, sensitivity 0.9976, specificity 0.9778, and accuracy 0.9957. The *PAX1*/*JAM3* baseline had substantially lower AUC in this TCGA setting (0.7225 ± 0.0251). At the evaluated threshold, it achieved sensitivity of 1.0000 but specificity of 0.0000, indicating poor calibration or threshold transfer in this dataset.

### External validation in GSE155760

External validation was performed using the TCGA-trained 10-gene logistic regression pipeline. The curated GSE155760 subset contained 33 endometrial cancer samples and 13 normal endometrial mucosa samples. All ten panel probes were present on the EPIC platform. The external AUC was 0.9790. At the default probability threshold of 0.5, sensitivity was 0.9091, specificity was 1.0000, accuracy was 0.9348, and F1 score was 0.9524. The confusion matrix contained 13 true negatives, 0 false positives, 3 false negatives, and 30 true positives.

### Comparison with *PAX1*/*JAM3* baseline

The 10-marker model substantially outperformed the *PAX1*/*JAM3* baseline in internal held-out comparison. DeLong testing showed a significant AUC difference, with 10-marker AUC 0.9966, baseline AUC 0.7115, AUC difference 0.2851, z = 6.2635, and p = 3.76 × 10^-10.

### Strength of the four data-driven markers

The four data-driven markers showed strong individual discrimination. *CYP26C1* had TCGA AUC 0.997 and GEO AUC 0.991. *SPARCL1* had TCGA AUC 0.986 and GEO AUC 0.981. *WDR52* had TCGA AUC 0.983 and GEO AUC 0.946. *CLDN15* had TCGA AUC 0.978 and GEO AUC 0.925. These data support their inclusion as candidate additions to established methylation markers, while recognizing that single-gene performance in array data does not replace targeted experimental validation.

### Batch-effect sensitivity analysis

PCA revealed some association between principal components and technical variables such as plate. ComBat reduced plate-associated structure, but tumor-normal separation remained strong after correction. This suggests that measured batch effects exist and should be reported, but the dominant tumor-normal methylation signal is not fully explained by available batch variables. Because batch and biology can be partially confounded in public datasets, external validation remains the most important safeguard against batch-driven overinterpretation.

## Discussion

This study developed a 10-gene promoter methylation panel for endometrial cancer detection by combining prior biological evidence with data-driven feature selection. The final panel retained genes with existing support in endometrial cancer or gynecologic methylation-marker literature while adding four data-driven genes that showed strong discrimination in both TCGA and an external GEO validation subset.

The most important methodological strength is leakage-aware validation. In high-dimensional omics studies, it is easy to obtain impressive performance by filtering features using the entire dataset before cross-validation. The fully nested design used here avoids that problem by repeating differential methylation analysis, promoter filtering, LASSO feature selection, and classifier training inside each outer training fold. The resulting AUC remained extremely high, suggesting that the signal distinguishing endometrial tumor from normal tissue is robust within TCGA-UCEC.

The second strength is external validation. The model trained on TCGA 450K data generalized to GSE155760 EPIC data, despite differences in platform, cohort composition, and data generation. The external AUC of 0.9790 is lower than the internal TCGA estimate, as expected, but still strong. The absence of false positives among 13 normal endometrial samples is encouraging, although the normal sample size is too small to precisely estimate clinical specificity.

The hybrid panel strategy also improves interpretability. The purely data-driven 10-gene panel performed similarly in TCGA but included genes that were harder to motivate biologically. By including *CDO1*, *PAX1*, *BHLHE22*, *HAND2*, *TBX5*, and *ZNF454*, the final panel becomes more suitable for experimental follow-up and manuscript framing. The four additional genes, *CYP26C1*, *SPARCL1*, *WDR52*, and *CLDN15*, provide strong data-driven signal and may represent useful additions to existing methylation marker combinations.

Several limitations should temper interpretation. First, both TCGA-UCEC and the GSE155760 subset are tissue-based public datasets. This means the results support tumor-versus-normal discrimination in tissue methylation data, but do not yet establish screening performance in asymptomatic populations. Second, the number of normal controls is limited, particularly in the external dataset. Third, batch effects are present and cannot be fully ruled out as contributors to model performance, even though nested validation, batch exploration, and external validation reduce this concern. Fourth, methylation-array probes are not identical to targeted assay amplicons; assay development must confirm that the selected CpG sites and nearby CpG regions can be robustly measured in clinically relevant specimens. Fifth, the default probability threshold was not clinically optimized and should not be interpreted as a deployable diagnostic cutoff.

The most direct next step is experimental validation. A practical follow-up study would design targeted methylation assays for the ten selected regions, test them first in independent tumor and benign/normal endometrial tissue samples, and then evaluate performance in minimally invasive specimens such as endometrial brushing or cervical brushing. The current panel should also be compared head-to-head against existing methylation-marker combinations, including *BHLHE22*/*CDO1*, *CDO1*/*CELF4*, and *PAX1*/*JAM3*-type assays, using the same specimens and thresholds.

## Conclusions

We identified a literature-constrained and data-driven 10-gene promoter methylation panel for endometrial cancer detection. The model achieved high performance in fully nested TCGA cross-validation and external validation in GSE155760. The findings are promising for biomarker development but remain computational and tissue-based. Further targeted assay validation in independent and minimally invasive clinical samples is required before claims about screening or clinical implementation can be made.

## Data availability

TCGA-UCEC methylation data are available through the Genomic Data Commons. GSE155760 is available through the Gene Expression Omnibus. Large raw and processed methylation matrices are not included in the repository because of size constraints.

## Code availability

Code, configuration files, selected processed results, manuscript materials, and reproducibility documentation are available at:

<https://github.com/wuwenbo0626/UCEC_methylation_biomarker>

## Ethics statement

This study used publicly available de-identified datasets and generated no new human participant data.

## Author contributions

W.W. conceived the project, implemented the computational workflow, performed analysis, interpreted results, organized the reproducible repository, and drafted the manuscript.

## Competing interests

The author declares no competing interests.

## Funding

No external funding was declared for this computational analysis.

## Figure legends

**Figure 1. Study workflow.** Overview of data acquisition, preprocessing, differential methylation analysis, promoter hypermethylation filtering, literature-constrained and LASSO-based marker selection, model training, internal validation, and external validation.

**Figure 2. Differential methylation volcano plot.** Genome-wide TCGA-UCEC differential methylation analysis comparing tumor and normal samples. Significant probes were defined using |delta beta| > 0.2 and FDR < 0.05.

**Figure 3. Final marker methylation levels.** Beta-value distributions for the ten final marker probes in TCGA-UCEC tumor and normal samples.

**Figure 4. ROC comparison.** ROC curves comparing the final 10-gene logistic regression model with the *PAX1*/*JAM3* baseline.

**Figure 5. Logistic regression coefficients.** Standardized coefficients for the final 10-gene logistic regression model.

**Figure 6. Confusion matrices.** Confusion matrix heatmaps for evaluated models.

## Main tables

### Table 1. Dataset overview

| Dataset | Platform | Tumor | Normal | Role |
|---|---|---:|---:|---|
| TCGA-UCEC | Illumina HumanMethylation450 | 420 | 46 | Discovery and internal validation |
| GSE155760 | Illumina EPIC | 33 | 13 | External validation |

### Table 2. Feature-screening summary

| Step | Count |
|---|---:|
| Cleaned TCGA probes | 390,516 |
| Cleaned TCGA samples | 466 |
| Significant differentially methylated probes | 50,919 |
| Tumor-hypermethylated DMPs | 21,414 |
| Tumor-hypomethylated DMPs | 29,505 |
| Promoter DMPs in TSS200/TSS1500 | 10,207 |
| Promoter hypermethylated probes | 4,848 |
| Deduplicated promoter hypermethylated candidate genes | 1,697 |

### Table 3. Final 10-gene panel

| Gene | Probe | Source | TCGA delta beta |
|---|---|---|---:|
| *CDO1* | cg23180938 | Literature-constrained | 0.5970 |
| *PAX1* | cg17620199 | Literature-constrained | 0.3136 |
| *BHLHE22* | cg06873806 | Literature-constrained | 0.3413 |
| *HAND2* | cg19178853 | Literature-constrained | 0.4158 |
| *TBX5* | cg06911121 | Literature-constrained | 0.4793 |
| *ZNF454* | cg24843380 | Literature-constrained | 0.6393 |
| *CYP26C1* | cg05219493 | Data-driven | 0.3935 |
| *SPARCL1* | cg08003102 | Data-driven | 0.5185 |
| *WDR52* | cg24199400 | Data-driven | 0.4138 |
| *CLDN15* | cg24809529 | Data-driven | 0.3772 |

### Table 4. Model performance summary

| Evaluation | AUC | Sensitivity | Specificity | Notes |
|---|---:|---:|---:|---|
| Fully nested TCGA 5-fold CV | 0.9989 ± 0.0024 | 0.9976 | 0.9778 | Feature discovery repeated inside training folds |
| Fixed 10-gene TCGA 5-fold CV | 0.9992 ± 0.0018 | 0.9976 | 0.9556 | Literature-constrained plus data-driven panel |
| Pure data-driven TCGA 5-fold CV | 0.9989 ± 0.0024 | 0.9976 | 0.9778 | Exploratory comparator |
| *PAX1*/*JAM3* TCGA baseline | 0.7225 ± 0.0251 | 1.0000 | 0.0000 | Baseline under evaluated threshold |
| GSE155760 external validation | 0.9790 | 0.9091 | 1.0000 | 33 tumor and 13 normal samples |

## References

References are provided in `manuscript/references.bib`. The marker-specific reference list should be manually checked once more before submission, especially for journal-specific formatting and whether review articles or primary endometrial cancer studies should be prioritized.
