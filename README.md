# UCEC methylation biomarker

一个可复现的 Python 项目，用于从 TCGA-UCEC 子宫内膜癌 Illumina HumanMethylation450K DNA 甲基化 beta 值数据中筛选候选筛查标志物，并完成差异甲基化分析、探针注释、候选基因筛选、功能富集、机器学习建模、PAX1/JAM3 baseline 对照和综合报告生成。

本项目按 Apple Silicon MacBook Pro（M2，16GB 内存）设计：大矩阵以 `.npy` 保存，表格以 `.parquet` 保存；不保存大 CSV；交叉验证只保存标量指标。

## 项目结构

```text
UCEC_methylation_biomarker/
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── docs/
│   ├── data_manifest.md
│   ├── plain_language_project_review.md
│   └── results_summary.md
├── models/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_differential_analysis.py
│   ├── 04_annotate_probes.py
│   ├── 05_enrichment_analysis.py
│   ├── 06_feature_selection.py
│   ├── 07_train_model.py
│   ├── 08_evaluate_model.py
│   ├── 09_assess_batch_effects_combat.py
│   ├── 10_nested_cv_lasso_lr_promoter_hyper.py
│   ├── 11_literature_constrained_10gene_panel.py
│   ├── 12_fully_nested_cv_dmp_lasso_lr.py
│   └── 13_external_validate_geo_gse155760.py
├── utils/
│   ├── data_utils.py
│   ├── plot_utils.py
│   └── model_utils.py
└── outputs/
    ├── figures/
    ├── reports/
    └── tables/
```

## 数据来源

- 项目：TCGA-UCEC
- 数据类型：DNA Methylation / Methylation Beta Value
- 芯片平台：Illumina Human Methylation 450
- 下载接口：GDC API
- 探针注释：Illumina HumanMethylation450 v1.2 manifest  
  `https://webdata.illumina.com/downloads/productfiles/humanmethylation450/humanmethylation450_15017482_v1-2.csv`

如果 Illumina manifest 下载失败，可使用 GEO GPL13534 备用文件：

`https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz`

## 环境安装

推荐在 macOS ARM / miniforge 中创建环境：

```bash
cd /Users/wuwenbo/AsiaInfo/UCEC_methylation_biomarker
conda create -n ucec_methylation python=3.11 -c conda-forge -y
conda activate ucec_methylation
python -m pip install -r requirements.txt
```

如果更偏好 conda：

```bash
mamba install -c conda-forge numpy pandas pyarrow scipy scikit-learn statsmodels matplotlib psutil requests tqdm pyyaml xgboost -y
python -m pip install gseapy
```

## 运行步骤

从零下载并分析：

```bash
python scripts/01_download_data.py
python scripts/02_preprocess.py
python scripts/03_differential_analysis.py
python scripts/04_annotate_probes.py
python scripts/05_enrichment_analysis.py
python scripts/06_feature_selection.py
python scripts/07_train_model.py
python scripts/08_evaluate_model.py
```

如果你已经有原始 `.npy` 和 `.parquet`，可以把文件放到 `data/processed/` 后从 `02_preprocess.py` 或 `03_differential_analysis.py` 开始运行。所有阈值、路径、模型参数均可在 `config.yaml` 修改。

## 脚本说明

| 脚本 | 功能 |
|---|---|
| `01_download_data.py` | 使用 GDC API 查询并下载 TCGA-UCEC 450K beta 文件，构建 beta 矩阵和样本/探针 metadata |
| `02_preprocess.py` | 根据 barcode 分组，过滤缺失、SNP、性染色体探针，并做 KNN 填补 |
| `03_differential_analysis.py` | beta 转 M 值，Tumor vs Normal t 检验，BH-FDR，生成火山图 |
| `04_annotate_probes.py` | 用 Illumina manifest 注释探针到基因和基因区域，筛选 TSS200/TSS1500 |
| `05_enrichment_analysis.py` | 筛选启动子区高甲基化候选基因，GO BP / KEGG 富集，已知标志物重叠 |
| `06_feature_selection.py` | LASSO CV、1-SE alpha、随机森林重要性排序，确定最终10标志物 |
| `07_train_model.py` | 训练 LR/RF/XGBoost/SVM，输出测试集指标、5折CV AUC、ROC和混淆矩阵 |
| `08_evaluate_model.py` | 构建 PAX1/JAM3 baseline，DeLong 检验，生成综合报告图表 |
| `09_assess_batch_effects_combat.py` | PCA 检查批次效应，并用 ComBat 做敏感性分析 |
| `10_nested_cv_lasso_lr_promoter_hyper.py` | 在固定启动子高甲基化特征池中做嵌套 LASSO+LR 交叉验证 |
| `11_literature_constrained_10gene_panel.py` | 构建文献约束 + 数据驱动 10 基因 panel，并与 baseline 比较 |
| `12_fully_nested_cv_dmp_lasso_lr.py` | 完全无偏嵌套 CV：每个外层训练折内重新做 DMP 筛选、启动子筛选、LASSO 和建模 |
| `13_external_validate_geo_gse155760.py` | 下载 GEO GSE155760，提取 panel 探针并进行外部验证 |

## 当前分析主要结果

本次已完成分析得到：

- 清洗后矩阵：390,516 探针 × 466 样本
- Tumor：420；Normal：46
- 差异甲基化探针：50,919
- 启动子区高甲基化探针：4,848
- 候选基因：1,697
- 主推最终 panel：文献约束 + 数据驱动 10 基因
  - CDO1 / cg23180938
  - PAX1 / cg17620199
  - BHLHE22 / cg06873806
  - HAND2 / cg19178853
  - TBX5 / cg06911121
  - ZNF454 / cg24843380
  - CYP26C1 / cg05219493
  - SPARCL1 / cg08003102
  - WDR52 / cg24199400
  - CLDN15 / cg24809529

完全无偏嵌套 CV：

- 外层 5 折；每个训练折内重新做 DMP 筛选、启动子筛选、LassoCV 和逻辑回归
- AUC：0.9989 ± 0.0024
- 敏感性：0.9976
- 特异性：0.9778

固定 panel 对比：

- 文献约束 + 数据驱动 10 基因 panel：AUC 0.9992 ± 0.0018
- 纯数据驱动 10 基因 panel：AUC 0.9989 ± 0.0024
- PAX1/JAM3 baseline：AUC 0.7225 ± 0.0251

GEO GSE155760 外部验证：

- Tumor：33；Normal：13
- 10 个 panel 探针全部在 EPIC 数据中找到
- AUC：0.9790
- 敏感性：0.9091
- 特异性：1.0000
- 混淆矩阵：TN/FP/FN/TP = 13/0/3/30

## 输出文件

主要输出位于：

- 中间结果：`data/results/`
- 表格结果：`outputs/tables/`
- 图表：`outputs/figures/`
- 报告：`outputs/reports/`
- 模型：`models/`
- 项目说明：`docs/`

所有大矩阵使用 `.npy`，所有结果表使用 `.parquet`。

更多结果说明见：

- `docs/results_summary.md`
- `docs/data_manifest.md`
- `docs/plain_language_project_review.md`

## 注意事项

1. TCGA Tumor/Normal 数量不平衡，建议最终模型必须进行外部 GEO 队列验证。
2. 当前模型性能来自组织样本，不等同于真实筛查样本性能。
3. `SVC(probability=True)` 在 sklearn 1.9 有未来弃用提示；如需长期维护，可改为 `CalibratedClassifierCV(SVC())`。
4. 富集分析依赖 Enrichr 在线服务，需要网络。
