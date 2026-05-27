import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
# =========================================================
# SHAP Utility Function for Logistic Regression
# =========================================================
def visualise_shap(
    model_paths,
    data_splits,
    split_key="X_test",
    max_display=10,
    save_plots=False,
    output_dir="shap_outputs",
    plots= "all"
):
    """
    Generate SHAP visualizations for multiple Logistic Regression models.
    Parameters
    ----------
    model_paths : dict
        Dictionary containing model names and saved joblib paths.
        Example:
        {
            "SMOTE_LogReg": "models/smote_lr.pkl",
            "ADASYN_LogReg": "models/adasyn_lr.pkl"
        }
    data_splits : dict
        Dictionary containing train/validation/test splits.
        Example structure:
        data_splits = {
            "SMOTE_LogReg": {
                "X_train": X_train_smote,
                "X_val": X_val_smote,
                "X_test": X_test_smote,
                "feature_columns": feature_list
            }
        }
    split_key : str
        Which split to use for SHAP visualization.
        Default = "X_test"
    max_display : int
        Maximum number of displayed features.
    save_plots : bool
        Whether to save generated plots.
    output_dir : str
        Directory to save plots.
    """
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # select the plots to visualise
    valid_plots = {"summary", "bar", "waterfall", "beeswarm"}

    if plots == "all":
        plots = valid_plots
    elif isinstance(plots, str):
        plots = {plots}
    else:
        plots = set(plots)

    for model_name, model_path in model_paths.items():
        print("=" * 60)
        print(f"Generating SHAP plots for: {model_name}")
        print("=" * 60)
        # Load saved model
        model = joblib.load(model_path)
        # Load selected dataset split
        X_test = data_splits[model_name][split_key]
        # Get feature names
        feature_names = data_splits[model_name]["feature_columns"]
        # Convert to DataFrame if needed
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
        # Create SHAP explainer
        explainer = shap.Explainer(model, X_test)
        # Calculate SHAP values
        shap_values = explainer(X_test)
    
        # =====================================================
        # 1. SHAP Summary Plot
        # =====================================================
        
        if "summary" in plots:
            shap.summary_plot(
                shap_values,
                X_test,
                max_display=max_display,
                show=False
            )
            plt.title(f"SHAP Summary Plot - {model_name}")
            if save_plots:
                plt.savefig(
                    f"{output_dir}/{model_name}_summary.png",
                    bbox_inches="tight",
                    dpi=300
                )
            plt.show()
        # =====================================================
        # 2. SHAP Bar Plot
        # =====================================================
        if "bar" in plots:
            shap.plots.bar(
                shap_values,
                max_display=max_display,
                show=False
            )
            plt.title(f"Logistic Regression - Feature {model_name}\n Feature Importance")
            if save_plots:
                plt.savefig(
                    f"{output_dir}/{model_name}_bar.png",
                    bbox_inches="tight",
                    dpi=300
                )
            plt.show()
        # =====================================================
        # 3. Waterfall Plot (First Sample)
        # =====================================================
        if "waterfall" in plots:
            shap.plots.waterfall(
                shap_values[0],
                max_display=max_display,
                show=True
            )
        # =====================================================
        # 4. Beeswarm Plot
        # =====================================================
        if "beeswarm" in plots:
            shap.plots.beeswarm(
                shap_values,
                max_display=max_display,
                show=True
            )

        print(f"Completed SHAP visualisation for: {model_name}\n")