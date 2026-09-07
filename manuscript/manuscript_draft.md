# A literature-constrained and data-driven DNA methylation panel for endometrial cancer detection using TCGA-UCEC and external GEO validation

**Authors:** Wuwenbo Wu  
**Correspondence:** Wuwenbo Wu  
**Manuscript type:** Computational biomarker discovery study  
**Draft status:** First internal draft; references and journal formatting require verification before submission.

## Abstract

### Background

Endometrial cancer is one of the most common gynecologic malignancies, and molecular biomarkers may improve early detection and risk stratification. DNA methylation is a promising biomarker modality because promoter hypermethylation can be stable, cancer-associated, and measurable by targeted assays. However, purely data-driven biomarker discovery can produce models that are difficult to interpret and vulnerable to overfitting, especially when feature selection is performed before validation.

### Methods

We developed a reproducible Python-based workflow for endometrial cancer methylation biomarker discovery using TCGA-UCEC Illumina HumanMethylation450 data. The discovery dataset included 420 tumor and 46 normal samples; after quality control, 390,516 probes across 466 samples were retained. Differential methylation analysis was performed using beta-to-M-value transformation, two-sample t-tests, Benjamini-Hochberg false discovery rate correction, and delta-beta estimation. Candidate features were restricted to promoter-region hypermethylated probes located in TSS200 or TSS1500 regions. We then constructed a literature-constrained and data-driven 10-gene panel by forcing inclusion of established or biologically supported markers and selecting additional features with LASSO. Model performance was evaluated using leakage-aware nested cross-validation in TCGA and external validation in GEO GSE155760. The final classifier was logistic regression.

### Results

The final panel included six literature-constrained genes (CDO1, PAX1, BHLHE22, HAND2, TBX5, and ZNF454) and four data-driven genes (CYP26C1, SPARCL1, WDR52, and CLDN15). Fully nested 5-fold cross-validation in TCGA, with differential methylation filtering and LASSO feature selection repeated inside each training fold, achieved an AUC of 0.9989 ± 0.0024, sensitivity of 0.9976, and specificity of 0.9778. The fixed literature-constrained 10-gene panel achieved an internal 5-fold AUC of 0.9992 ± 0.0018, sensitivity of 0.9976, and specificity of 0.9556. External validation in GSE155760, including 33 tumor and 13 normal endometrial samples, yielded an AUC of 0.9790, sensitivity of 0.9091, and specificity of 1.0000. Compared with a PAX1/JAM3 two-gene baseline, the 10-marker model showed substantially higher AUC in internal comparison, with DeLong p = 3.76 × 10^-10.

### Conclusions

This study identifies a biologically interpretable 10-gene promoter methylation panel for endometrial cancer detection and demonstrates strong internal and external performance in public tissue-based methylation-array cohorts. The results support further experimental validation using targeted methylation assays and clinically relevant sample types such as endometrial brushing, cervical brushing, or other minimally invasive specimens. The current findings should be interpreted as computational biomarker evidence rather than clinical screening validation.

## Keywords

Endometrial cancer; DNA methylation; TCGA-UCEC; biomarker discovery; LASSO; logistic regression; external validation; promoter hypermethylation.

## Introduction

Endometrial cancer is a major malignancy of the female reproductive tract. Although many patients present with abnormal uterine bleeding, clinically deployable molecular markers could help improve early detection, diagnostic triage, and future risk-adapted screening strategies. Among molecular biomarker modalities, DNA methylation is especially attractive because aberrant promoter methylation is a common epigenetic feature of cancer and can be detected using targeted assays compatible with clinical specimens.

Several methylation markers have been studied in gynecologic cancers, including genes such as PAX1, JAM3, CDO1, HAND2, and other promoter-hypermethylated loci. These prior studies suggest that methylation signals can separate malignant from non-malignant endometrial tissue. However, translating genome-wide methylation discovery into a practical marker panel remains challenging. First, high-dimensional methylation-array data contain hundreds of thousands of probes but relatively few normal controls. Second, if feature selection is conducted on the full dataset before train-test splitting, downstream model performance can be severely inflated. Third, purely data-driven markers may include genes with unclear biological relevance, making clinical or experimental follow-up less compelling.

To address these issues, we designed a literature-constrained and data-driven biomarker discovery workflow. The analysis began with TCGA-UCEC HumanMethylation450 data and focused on promoter-region hypermethylated probes, because such loci are biologically plausible candidates for targeted methylation assays. We combined forced inclusion of known or literature-supported genes with LASSO-based feature selection, then evaluated model performance using both fully nested cross-validation and external validation in an independent GEO cohort. The goal was not to claim immediate clinical readiness, but to generate a transparent, reproducible, and experimentally actionable 10-gene methylation panel for future validation.

## Methods

### Study design

This was a retrospective computational biomarker discovery and validation study using publicly available DNA methylation-array datasets. TCGA-UCEC was used for discovery and internal validation. GEO GSE155760 was used as an external validation cohort. All analyses were implemented in Python and organized into a reproducible GitHub repository.

### Discovery dataset

The discovery dataset was TCGA-UCEC DNA methylation data profiled on the Illumina HumanMethylation450 array. The final analyzed dataset included 420 primary tumor samples and 46 solid tissue normal samples. After preprocessing and quality control, the methylation beta-value matrix contained 390,516 probes and 466 samples.

### External validation dataset

The external validation dataset was GSE155760. The analyzed subset included 33 endometrial cancer samples and 13 normal endometrial samples. Samples were profiled on the Illumina EPIC methylation array. All ten probes in the final panel were present in this external dataset.

### Methylation data preprocessing

Raw or downloaded methylation beta-value matrices were processed using Python with pandas and numpy. Samples were assigned to tumor or normal groups based on TCGA barcode-derived sample type information. Probes with excessive missingness were removed, samples with excessive missingness were excluded, SNP probes beginning with `rs` were filtered, and probes on sex chromosomes were removed. Remaining missing values were imputed using k-nearest-neighbor imputation. Processed matrices were stored as `.npy` arrays, and sample metadata were stored as `.parquet` files to avoid inefficient large CSV outputs.

### Differential methylation analysis

For statistical testing, beta values were converted to M values using:

```text
M = log2(beta / (1 - beta))
```

Beta values equal to 0 or 1 were clipped to 1e-6 and 0.999999 before transformation. For each probe, tumor and normal groups were compared using two-sample t-tests on M values. Delta beta was calculated as the difference between mean tumor beta and mean normal beta:

```text
delta beta = mean(beta_tumor) - mean(beta_normal)
```

P values were adjusted using the Benjamini-Hochberg false discovery rate method. Differentially methylated probes were initially defined as probes with absolute delta beta greater than 0.2 and FDR less than 0.05.

### Probe annotation and promoter restriction

HumanMethylation450 probes were annotated using Illumina manifest-derived annotation. Candidate probes were restricted to promoter-associated regions, defined as TSS200 or TSS1500. Because the intended translational direction was methylation-based cancer detection, the main candidate pool prioritized tumor hypermethylated promoter probes with positive delta beta.

### Literature-constrained and data-driven panel construction

The final panel was constructed using a hybrid strategy. Six genes with prior biological or biomarker support were forced into the panel when eligible promoter-region hypermethylated probes were available: CDO1, PAX1, BHLHE22, HAND2, TBX5, and ZNF454. For each forced gene, the representative probe was selected from promoter-region hypermethylated probes, prioritizing high methylation variance across samples. Additional data-driven genes were selected from the remaining candidate pool using LASSO-based feature selection. Pseudogenes, genes with unclear functional annotation, and previously selected biologically ambiguous candidates such as CD8A, HIST1H4F, and PCDHB19P were excluded from this panel-building step.

The final 10-gene panel consisted of:

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

All selected markers were hypermethylated in tumor samples.

### Model training

The final model was logistic regression using the ten selected beta-value features. Logistic regression was prioritized because its performance was comparable to more complex models in internal analyses while retaining interpretability and ease of future clinical translation. Candidate models evaluated during development included logistic regression, random forest, XGBoost, and support vector machines.

### Leakage-aware internal validation

To estimate performance without feature-selection leakage, we implemented fully nested 5-fold stratified cross-validation. In each outer training fold, differential methylation analysis was recomputed using only the training samples, promoter-region hypermethylated probes were reselected, LASSO feature selection was performed inside the training data, and logistic regression was trained using the selected features. The trained model was then evaluated on the held-out outer validation fold. Metrics included AUC, sensitivity, specificity, accuracy, and F1 score.

### Fixed-panel comparison

The final literature-constrained 10-gene panel was compared with a pure data-driven 10-gene panel and a PAX1/JAM3 two-gene baseline using the same 5-fold cross-validation framework where applicable. For the PAX1/JAM3 baseline, representative promoter-region probes were selected and logistic regression was trained using only those two features.

### External validation

For external validation, the TCGA-trained logistic regression pipeline was applied to GSE155760. The ten panel probes were extracted from the EPIC methylation matrix. Prediction performance was evaluated using AUC, sensitivity, specificity, accuracy, F1 score, and confusion matrix values at the default probability threshold of 0.5.

### Batch-effect assessment

Potential batch variables were explored from available metadata, including sample type and technical variables such as plate when available. Principal component analysis was used to visualize sample clustering by biological group and batch variables. ComBat-based correction was explored as a sensitivity analysis. Batch-associated structure was observed, but tumor-normal separation remained strong after correction, supporting that the classification signal was not fully explained by measured batch effects.

### Statistical analysis and software

Analyses were conducted in Python 3.11. Core packages included numpy, pandas, scipy, statsmodels, scikit-learn, pyarrow, matplotlib, seaborn, xgboost, gseapy, and joblib. Random seeds were fixed where applicable. Random forest and cross-validation routines used limited parallelization to fit execution on a MacBook Pro with Apple M2 and 16 GB memory. Large matrices were stored as `.npy` or `.parquet` files rather than CSV.

## Results

### Dataset preprocessing

The TCGA-UCEC methylation dataset included 420 tumor and 46 normal samples. After quality control, the cleaned matrix retained 390,516 probes across 466 samples. This matrix was used for differential methylation analysis, candidate marker screening, and internal validation.

### Promoter hypermethylation candidate discovery

Differential methylation analysis identified a large set of tumor-associated methylation changes. Because promoter hypermethylation is a biologically plausible mechanism for gene silencing and targeted assay development, downstream marker selection focused on TSS200/TSS1500 promoter probes with positive delta beta. This strategy yielded a candidate pool suitable for both data-driven modeling and literature-constrained marker prioritization.

### Final 10-gene panel

The final 10-gene panel included CDO1, PAX1, BHLHE22, HAND2, TBX5, ZNF454, CYP26C1, SPARCL1, WDR52, and CLDN15. Six genes were included based on literature or biological support, and four genes were selected by data-driven analysis. All ten representative probes were promoter-associated and tumor-hypermethylated in TCGA-UCEC.

### Internal validation using fully nested cross-validation

Fully nested 5-fold cross-validation was used to assess whether high model performance was driven by information leakage. In each outer fold, the full feature-discovery process was repeated using only training data. The nested evaluation achieved an AUC of 0.9989 ± 0.0024, mean sensitivity of 0.9976, mean specificity of 0.9778, and mean accuracy of 0.9957. These results suggest that the tumor-normal methylation signal is robust within TCGA-UCEC and is not mainly attributable to feature-selection leakage.

### Fixed-panel comparison

Using a fixed 10-gene panel, the literature-constrained plus data-driven model achieved an AUC of 0.9992 ± 0.0018, sensitivity of 0.9976, and specificity of 0.9556 in TCGA 5-fold cross-validation. A pure data-driven 10-gene panel showed similar internal AUC (0.9989 ± 0.0024), sensitivity (0.9976), and specificity (0.9778), but contained several genes with less direct translational interpretability. The PAX1/JAM3 baseline achieved lower discrimination, with AUC 0.7225 ± 0.0251, sensitivity 1.0000, and specificity 0.0000 under the evaluated threshold. DeLong comparison of the 10-marker model and PAX1/JAM3 baseline showed a statistically significant AUC difference (p = 3.76 × 10^-10).

### External validation in GSE155760

The TCGA-trained 10-gene logistic regression model was externally validated in GSE155760. The validation subset included 33 endometrial cancer samples and 13 normal endometrial samples. All ten panel probes were available on the EPIC array. The model achieved an AUC of 0.9790, sensitivity of 0.9091, specificity of 1.0000, accuracy of 0.9348, and F1 score of 0.9524. At the default threshold of 0.5, the confusion matrix was 13 true negatives, 0 false positives, 3 false negatives, and 30 true positives.

### Individual strength of data-driven markers

The four newly prioritized data-driven markers showed strong individual discrimination. CYP26C1 achieved TCGA AUC 0.997 and GEO AUC 0.991. SPARCL1 achieved TCGA AUC 0.986 and GEO AUC 0.981. WDR52 achieved TCGA AUC 0.983 and GEO AUC 0.946. CLDN15 achieved TCGA AUC 0.978 and GEO AUC 0.925. These results support the inclusion of these genes alongside known methylation markers, although experimental validation remains necessary.

### Batch-effect sensitivity analysis

PCA revealed partial association between principal components and technical variables such as plate. ComBat correction reduced plate-associated structure, while tumor-normal separation remained strong. This indicates that batch effects are present and should be transparently reported, but the dominant methylation signal separating tumor from normal samples is unlikely to be explained solely by measured batch variables. External validation in GSE155760 further supports generalizability across datasets and platforms.

## Discussion

This study developed a 10-gene DNA methylation panel for endometrial cancer detection using a workflow designed to balance interpretability and predictive performance. The final panel combines known or biologically supported methylation markers with additional data-driven candidates that showed strong statistical and predictive signals in both TCGA and GEO data.

One strength of the study is the explicit control of feature-selection leakage. Initial biomarker analyses can easily overestimate performance if the full dataset is used to select features before validation. Here, the nested cross-validation analysis repeated differential methylation testing, promoter filtering, LASSO feature selection, and model fitting entirely within the training folds. The resulting performance remained high, suggesting that the methylation distinction between tumor and normal endometrial tissue is robust in the analyzed public datasets.

Another strength is external validation. The model trained on TCGA 450K data performed well in GSE155760 EPIC data, and all ten probes were available across platforms. The external AUC decreased relative to internal TCGA estimates, which is expected when moving across datasets, laboratories, platforms, and sample compositions. Nevertheless, the external AUC remained high, and the model produced no false positives among 13 normal endometrial samples at the default threshold. Because the normal external sample size is small, this specificity estimate should be interpreted cautiously.

The hybrid panel-building strategy also improved biological plausibility. A purely data-driven model achieved very high internal performance but included some genes with unclear translational relevance. By enforcing inclusion of CDO1, PAX1, BHLHE22, HAND2, TBX5, and ZNF454, while selecting CYP26C1, SPARCL1, WDR52, and CLDN15 from the data, the final panel preserved strong performance while providing a clearer rationale for downstream assay development.

The study has several limitations. First, both TCGA-UCEC and GSE155760 are public tissue-based methylation-array datasets. These data do not establish clinical screening performance in asymptomatic populations or minimally invasive sample types. Second, the number of normal samples is limited, particularly in the external cohort. Third, batch effects and dataset-level differences exist and cannot be fully excluded by computational correction. Fourth, methylation-array probes may not map perfectly to targeted assay amplicons, so wet-lab assay design must validate the exact CpG sites and nearby CpG regions. Finally, model thresholds were not optimized in a prospective clinical context.

Future work should validate the 10-gene panel using targeted bisulfite sequencing, pyrosequencing, methylation-specific PCR, or another clinically feasible methylation assay. The most important next step is testing the panel in independent clinical specimens, ideally including endometrial brushing, cervical brushing, or other minimally invasive samples. A head-to-head comparison with existing marker combinations, including PAX1/JAM3 and CDO1-based assays, would further clarify translational value.

## Conclusions

We identified a literature-constrained and data-driven 10-gene promoter methylation panel for endometrial cancer detection. The panel showed strong internal performance in leakage-aware nested cross-validation and high external validation performance in GSE155760. These findings support further experimental and clinical validation, while emphasizing that current evidence is computational and tissue-based rather than definitive clinical screening validation.

## Data availability

TCGA-UCEC methylation data are publicly available through the Genomic Data Commons. GSE155760 is publicly available through the Gene Expression Omnibus. Processed large methylation matrices are not included in this repository because of file size; scripts are provided to regenerate processed data where access and storage are available.

## Code availability

The reproducible project repository is available at:

https://github.com/wuwenbo0626/UCEC_methylation_biomarker

## Ethics statement

This study used publicly available de-identified datasets. No new human participant data were generated for this computational analysis.

## Figure legends

**Figure 1. Study workflow.** Overview of TCGA-UCEC methylation data acquisition, preprocessing, differential methylation analysis, promoter hypermethylation filtering, literature-constrained and LASSO-based marker selection, model training, internal validation, and external validation.

**Figure 2. Differential methylation volcano plot.** Genome-wide differential methylation analysis comparing TCGA-UCEC tumor and normal samples, showing delta beta and statistical significance after FDR correction.

**Figure 3. Methylation levels of final candidate markers.** Boxplots showing beta-value distributions for the ten final panel probes in tumor and normal samples.

**Figure 4. ROC comparison of the final 10-gene model and PAX1/JAM3 baseline.** Receiver operating characteristic curves comparing model discrimination.

**Figure 5. Logistic regression feature coefficients.** Coefficients of the final logistic regression model using the ten selected methylation markers.

**Figure 6. Confusion matrices.** Classification results for evaluated models, shown as confusion matrix heatmaps.

## Tables

**Table 1. Dataset overview.**

| Dataset | Platform | Tumor | Normal | Role |
|---|---|---:|---:|---|
| TCGA-UCEC | Illumina HumanMethylation450 | 420 | 46 | Discovery and internal validation |
| GSE155760 | Illumina EPIC | 33 | 13 | External validation |

**Table 2. Final 10-gene panel.**

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

**Table 3. Internal and external validation performance.**

| Evaluation | AUC | Sensitivity | Specificity | Notes |
|---|---:|---:|---:|---|
| Fully nested TCGA 5-fold CV | 0.9989 ± 0.0024 | 0.9976 | 0.9778 | Feature discovery repeated inside outer folds |
| Fixed 10-gene TCGA 5-fold CV | 0.9992 ± 0.0018 | 0.9976 | 0.9556 | Literature-constrained plus data-driven panel |
| GSE155760 external validation | 0.9790 | 0.9091 | 1.0000 | 33 tumor, 13 normal |
| PAX1/JAM3 TCGA baseline | 0.7225 ± 0.0251 | 1.0000 | 0.0000 | Two-gene baseline under evaluated threshold |

## References

> Draft reference list. Verify all metadata, titles, author lists, journal names, page numbers, and DOIs before submission.

1. The Cancer Genome Atlas Research Network. Integrated genomic characterization of endometrial carcinoma. *Nature*. 2013.
2. Grossman RL, Heath AP, Ferretti V, et al. Toward a shared vision for cancer genomic data. *New England Journal of Medicine*. 2016.  
3. Colaprico A, Silva TC, Olsen C, et al. TCGAbiolinks: an R/Bioconductor package for integrative analysis of TCGA data. *Nucleic Acids Research*. 2016.
4. Leek JT, Johnson WE, Parker HS, Jaffe AE, Storey JD. The sva package for removing batch effects and other unwanted variation in high-throughput experiments. *Bioinformatics*. 2012.
5. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*. 2007.
6. Tibshirani R. Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*. 1996.
7. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*. 1995.
8. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves. *Biometrics*. 1988.
9. Gene Expression Omnibus accession GSE155760. National Center for Biotechnology Information.
10. Literature on PAX1/JAM3 methylation markers in endometrial cancer. To be verified and replaced with exact primary citations.
11. Literature on CDO1 methylation in endometrial cancer. To be verified and replaced with exact primary citations.
12. Literature on HAND2 methylation in endometrial cancer. To be verified and replaced with exact primary citations.

