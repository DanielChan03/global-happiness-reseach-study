# WVS Happiness Prediction and Interpretability Study

This repository contains a machine learning research workflow for predicting and interpreting happiness using the World Values Survey (WVS) Wave 7 cross-national dataset. The project cleans WVS survey responses, engineers theory-guided predictors, compares machine learning and deep learning models, applies explainable AI methods, and constructs an interpretable logistic-regression happiness formula from a compact feature set.

The analysis is organized as sequential notebooks from `01` to `08`, with paired no-resampling and SMOTE branches for model evaluation.

## Research Aim

The study investigates whether happiness can be predicted from WVS survey responses and which social, psychological, civic, ethical, and demographic factors contribute most strongly to the prediction.

The workflow focuses on:

- cleaning and recoding WVS Wave 7 survey data from 2017-2022;
- mapping questionnaire variables into interpretable feature names;
- reducing multicollinearity and feature redundancy;
- comparing original and SMOTE-balanced training settings;
- benchmarking classical ML models and PyTorch neural networks;
- explaining tree-based and logistic models with SHAP;
- building a compact, interpretable happiness probability formula.

## Repository Structure

```text
.
|-- data/
|   |-- raw/                         # WVS raw data, questionnaire, country codes
|   |-- processed/                   # cleaned and feature-engineered datasets
|   |-- train_sets/                  # original and SMOTE training sets
|   |-- val_sets/                    # validation sets
|   |-- test_sets/                   # test sets
|   |-- DL_hyperparameter_results/   # deep learning tuning results
|   `-- feature_sets_performance/    # feature-combination model reports
|-- data_documentation/              # WVS codebook and documentation PDFs
|-- figures/                         # pipeline, SHAP, ROC, and interpretation figures
|-- model/                           # saved model artifacts
|-- notebooks/                       # full research workflow, ordered 01-08
|-- publication/                     # publication manuscript PDF
|-- report/                          # final project report PDF
|-- research_papers/                 # related reference papers
`-- src/my_ml_tools/                 # reusable helper functions
```

## Notebook Workflow

Run the notebooks in this order.

| Step | Notebook | Purpose |
|---|---|---|
| 01 | `notebooks/01_data_cleaning.ipynb` | Loads WVS Wave 7 data, handles WVS missing-value codes, filters to 2017-2022, removes irrelevant/high-missing variables, maps questionnaire names, validates ranges, recodes variables, reverses scales, compresses scales, and exports `cleaned_df.csv`. |
| 02 | `notebooks/02_eda.ipynb` | Performs exploratory analysis, maps country codes, visualizes demographic and happiness distributions, and explores theoretical feature groups using stacked bar charts and Spearman correlations. |
| 03 | `notebooks/03_feature_engineering.ipynb` | Drops redundant or bias-prone variables, applies VIF feature selection, imputes missing values, encodes/scales features, runs RFECV, converts happiness into a binary target, creates train/validation/test splits, and generates original plus SMOTE training sets. |
| 04a | `notebooks/04_ml_model_training_evaluation/04a_ml_no_resampling.ipynb` | Tunes and evaluates Random Forest, Gradient Boosting, AdaBoost, XGBoost, LightGBM, CatBoost, and Extra Trees on the original training data. |
| 04b | `notebooks/04_ml_model_training_evaluation/04b_ml_smote.ipynb` | Repeats the classical ML benchmark using the SMOTE-resampled training data. |
| 05 | `notebooks/05_dl_model_training_evaluation/` | Trains and compares PyTorch `SimpleMLP` and `DCN` models under original and SMOTE settings. |
| 06 | `notebooks/06_xai_analysis/` | Uses SHAP to explain the saved CatBoost no-resampling model and LightGBM SMOTE model, including global feature importance and income-stratified SHAP analysis. |
| 07 | `notebooks/07_further_feature_combination_evaluation/` | Evaluates theory-driven feature combinations under original and SMOTE settings using Random Forest, boosted models, Logistic Regression, and Naive Bayes. |
| 08 | `notebooks/08_feature_interpretability_formula_construct.ipynb` | Builds the compact logistic-regression happiness formula, calculates SHAP feature weights, runs Wald and likelihood-ratio tests, and demonstrates example probability/risk cases. |

## Data and Target

The project uses WVS Wave 7 cross-national survey data:

- raw survey file: `data/raw/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv.zip`
- questionnaire mapping: `data/raw/Questionnaire.xlsx`
- country codes: `data/raw/country_code.csv`
- WVS documentation: `data_documentation/`

The target variable is `happy`. During feature engineering, the original happiness response is label-encoded and reduced into a binary classification target:

- `0`: lower happiness classes
- `1`: higher happiness classes

The modeling split is stratified into train, validation, and test sets. SMOTE is applied only to the training set.

## Feature Themes

The cleaned variables are organized around several conceptual groups, including:

- self-actualization: `sact`
- physiological conditions: `physio*`
- safety: `safety*`
- belonging: `belonging*`
- esteem: `esteem*`
- trust: `trust*`
- ethical values: `ethical*`
- civic values: `civic*`
- compassion: `compassion*`
- growth: `growth*`
- socioeconomic and demographic variables such as `age`, `income_scale`, `marital_status`, and `religious1`

The final compact hybrid formula in Notebook 08 uses Combination 5:

```text
sact, physio1, safety5, trust3, ethical80
```

## Saved Results

Key saved artifacts include:

- processed modeling data: `data/processed/df_fea_eng.csv`
- final train/test/validation sets: `data/train_sets/`, `data/test_sets/`, `data/val_sets/`
- model comparison reports: `data/feature_sets_performance/`
- deep learning tuning results: `data/DL_hyperparameter_results/`
- SHAP plots: `figures/ml_evaluation_shap/` and `figures/logistic_shap/`
- ROC curves: `figures/roc/`
- saved models: `model/`

Headline results from the saved reports:

| Setting | Best saved model/result | Test accuracy | Test MCC |
|---|---:|---:|---:|
| Full feature set, no resampling | CatBoost | 0.8785 | 0.3977 |
| Full feature set, SMOTE | LightGBM | 0.8779 | 0.3966 |
| Compact Combination 5, no resampling | Logistic Regression | 0.8759 | 0.3692 |
| Compact Combination 5, SMOTE | Random Forest | 0.8715 | 0.3822 |
| Deep learning, no resampling | SimpleMLP | 0.8772 | not reported in CSV |
| Deep learning, SMOTE | DCN | 0.8603 | not reported in CSV |

The compact Combination 5 gives competitive performance with only five predictors, making it useful for the final interpretable happiness formula.

## Local Setup

The notebooks were written for Google Colab and assume:

```python
DRIVE_DATA_PATH = "/content/drive/My Drive/WVS_happiness_study"
```

For local execution, either update `DRIVE_DATA_PATH` in the notebooks or adapt the paths to this repository root.

Recommended local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn statsmodels scipy joblib tqdm openpyxl xgboost lightgbm catboost shap torch
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Then open the notebooks in Jupyter or VS Code and run them sequentially.

## Reproducibility Notes

- The notebooks use `random_state=42` or equivalent seeds where practical.
- Train/validation/test splitting is stratified.
- Imputation and preprocessing are fitted on the training workflow to reduce leakage.
- SMOTE is applied to training data only.
- Tree-based model explanations are generated with `shap.TreeExplainer`.
- Logistic-regression formula interpretation uses fitted model coefficients, while SHAP weights are used for contribution analysis.

## Helper Modules

Reusable code lives in `src/my_ml_tools/`:

| Module | Description |
|---|---|
| `data_io.py` | Generic load/export helpers for CSV, Excel, JSON, and Parquet files. |
| `feature_engineering.py` | VIF feature-selection utilities and class-imbalance experiment helpers. |
| `training.py` | Shared training/evaluation function for tuned and untuned scikit-learn models. |
| `evaluation.py` | SHAP visualization and normalized SHAP-weight utilities for logistic models. |

## Interpretation and Limitations

This is a predictive and interpretability study, not a causal analysis. The WVS responses are self-reported survey variables, and model coefficients or SHAP values should be interpreted as associations within the fitted data and preprocessing pipeline. Some variables, such as country, race, birth country, and spouse-related variables, are removed during feature engineering to reduce noise, redundancy, or bias risk.

Before redistributing raw WVS data, check the World Values Survey terms of use.

## Main Outputs

For the full write-up, see:

- final report: `report/1231302741_NeohHoo-Thye_FYP2_Report.pdf`
- publication manuscript: `publication/happiness_study_AIPCP.pdf`
