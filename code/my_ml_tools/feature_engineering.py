## 1. Variation Inflation Analysis functions
import builtins
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.stats.outliers_influence import variance_inflation_factor

def inter_vif(df, cols, target, threshold=10.0):
    """
    Perform iterative inter-group Variance Inflation Factor (VIF) analysis.

    This function identifies variables with high multicollinearity within
    a given feature group by repeatedly selecting the variable with the
    highest VIF value above the specified threshold.

    Workflow:
    ---------
    1. Compute VIF values for all variables in the group.
    2. Select the variable with the highest VIF.
    3. If the highest VIF exceeds the threshold:
           - Keep the variable as a representative high-collinearity feature.
           - Remove it from further VIF computation.
    4. Repeat until:
           - No variables exceed the threshold, or
           - Only one variable remains.
    5. Among all retained high-VIF variables, determine the "winner"
       using the strongest absolute correlation with the target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing all variables.

    cols : list
        List of feature column names to evaluate for inter-group
        multicollinearity.

    target : str
        Name of the target column used to determine the strongest
        representative variable.

    threshold : float, default=10.0
        VIF threshold used to determine high multicollinearity.

    Returns:
    --------
    winner : str or None
        Feature with the strongest absolute correlation to the target
        among the retained high-VIF variables.

    survivors : list
        List of retained high-VIF variables.

    no_multi_grp_vars : list
        Variables belonging to groups where no feature exceeded the
        VIF threshold.

    dropped : list of tuples
        Variables removed because their VIF values were below the
        threshold during the final iteration.

        Format:
            [(feature_name, vif_value), ...]

    Notes:
    ------
    - Uses parallel processing via joblib for faster VIF computation.
    - Designed for grouped feature selection workflows.
    - Useful for identifying representative variables among highly
      correlated feature groups.
    """
    
    # If range is not callable, restore it
    if not callable(range):
        print("Restoring range function...")
        range = builtins.range
        len = builtins.len

    df_use = df[cols].dropna().copy()
    temp_vars = cols.copy()

    kept_vars = []      # store HIGH VIF variables (VIF > 10)
    dropped = []        # store LOW VIF variables (≤ 10)
    no_multi_grp_vars = []  # store variables from group with no high-VIF

    print(f"\n🌟 Processing group with {len(cols)} variables:", cols)

    pbar = tqdm(desc=f"VIF loop ({cols[0].split('_')[0]})",
                total=len(cols), leave=False)

    while True:
        X = df_use[temp_vars].astype(np.float32).values

        # Compute VIF for all remaining vars
        vif_list = Parallel(n_jobs=-1)(
            delayed(variance_inflation_factor)(X, i)
            for i in range(X.shape[1])
        )

        vif_df = pd.DataFrame({"Feature": temp_vars, "VIF": vif_list})
        vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

        max_vif = vif_df.iloc[0]["VIF"]
        worst = vif_df.iloc[0]["Feature"]

        # If highest VIF is HIGH → KEEP it
        if max_vif > threshold:
            kept_vars.append(worst)
            print(f"  → Added {worst} (VIF={max_vif:.2f})")

            temp_vars.remove(worst)     # remove from pool
            pbar.update(1)

        else:
            # All remaining variables are LOW VIF → drop them all
            for feat, v in zip(vif_df["Feature"], vif_df["VIF"]):
                dropped.append((feat, v))
                print(f"  → Dropped {feat} (VIF={v:.2f})")

            # If no high-VIF variable found at all, mark all as no_multicollinearity
            if len(kept_vars) == 0:
                no_multi_grp_vars.extend(temp_vars)
            break

        # Stop if only 1 variable left (no more VIF possible)
        if len(temp_vars) <= 1:
            break

    pbar.close()

    # Determine the winner among high-VIF variables (best correlation with target)
    if len(kept_vars) > 1:
        corrs = df[kept_vars].corrwith(df[target]).abs()
        winner = corrs.idxmax()
        winner_corr = corrs[winner]
    elif len(kept_vars) == 1:
        winner = kept_vars[0]
        winner_corr = df[winner].corr(df[target])
    else:
        winner = None
        winner_corr = None

    survivors = kept_vars  # all high-VIF variables

    if winner is not None:
        print(f"\nWinner: {winner} (|corr| = {winner_corr:.4f})")
    else:
        print("\nWinner: None")


    if no_multi_grp_vars:
        print(f"No-multicollinearity variables: {no_multi_grp_vars}")
    else:
        print(f"Survivors (high-VIF): {survivors}")

    return winner, survivors, no_multi_grp_vars, dropped


def intra_vif(df, cols, threshold=10.0):
    """
    Perform iterative intra-group Variance Inflation Factor (VIF) analysis.

    This function removes highly collinear variables within a feature group
    by repeatedly dropping the variable with the highest VIF value until
    all remaining variables fall below the specified threshold.

    Workflow:
    ---------
    1. Compute VIF values for all variables in the group.
    2. Identify the variable with the highest VIF.
    3. If the highest VIF exceeds the threshold:
           - Remove the variable from the group.
    4. Repeat until:
           - All remaining variables have VIF ≤ threshold, or
           - Only one variable remains.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing all variables.

    cols : list
        List of feature column names to evaluate for intra-group
        multicollinearity.

    threshold : float, default=10.0
        VIF threshold used to determine high multicollinearity.

    Returns:
    --------
    kept_vars : list
        Remaining variables after iterative VIF filtering.

    dropped : list
        Variables removed due to high multicollinearity.

    Notes:
    ------
    - Uses parallel processing via joblib for faster VIF computation.
    - Designed for removing redundant variables within the same feature group.
    - Useful for reducing multicollinearity before machine learning modeling.
    """

        # If range is not callable, restore it
    if not callable(range):
        print("Restoring range function...")
        range = builtins.range
        len = builtins.len
        
    df_use = df[cols].dropna().copy()
    temp_vars = cols.copy()

    kept_vars = []   # variables to keep
    dropped = []     # variables to drop due to high VIF

    print(f"\n🌟 Processing intra-VIF for {len(cols)} variables:", cols)

    pbar = tqdm(total=len(cols), desc="Intra-VIF loop", leave=False)

    while True:
        X = df_use[temp_vars].astype(np.float32).values

        # Compute VIF for all remaining variables
        vif_list = Parallel(n_jobs=-1)(
            delayed(variance_inflation_factor)(X, i) for i in range(X.shape[1])
        )

        vif_df = pd.DataFrame({"Feature": temp_vars, "VIF": vif_list})
        vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

        max_vif = vif_df.iloc[0]["VIF"]
        worst = vif_df.iloc[0]["Feature"]

        if max_vif > threshold:
            # Drop the variable with highest VIF
            dropped.append(worst)
            temp_vars.remove(worst)
            print(f"  → Dropped {worst} (VIF={max_vif:.2f})")
        else:
            # All remaining variables are below threshold → keep them
            kept_vars.extend(temp_vars)
            break

        if len(temp_vars) <= 1:
            kept_vars.extend(temp_vars)
            break

        pbar.update(1)

    pbar.close()

    print(f"\nRemaining variables after intra-VIF: {kept_vars}")

    return kept_vars, dropped


## 2. Class Imbalance Analysis
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_imbalance_experiment(
    sampler,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model=None,
    name="method"
):
    """
    Run a class imbalance experiment using a resampling technique
    and evaluate model performance.

    Parameters:
    -----------
    sampler : imblearn sampler object or None
        Resampling technique such as SMOTE, RandomUnderSampler, etc.
        If None, no resampling is applied.

    X_train, y_train : training dataset

    X_val, y_val : validation dataset

    X_test, y_test : testing dataset

    model : sklearn-compatible estimator, optional
        Classification model used for training.
        Defaults to RandomForestClassifier.

    name : str, default="method"
        Name of the imbalance handling method.

    Returns:
    --------
    results : dict
        Dictionary containing:
            - method
            - X_resampled
            - y_resampled
            - y_val_pred
            - y_test_pred
            - val_accuracy
            - test_accuracy
    """

    # Default model
    if model is None:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # IMPORTANT:
    # create fresh independent model instance
    model = clone(model)

    # =========================
    # 1. Resampling
    # =========================
    if sampler is None:
        X_res, y_res = X_train, y_train
    else:
        X_res, y_res = sampler.fit_resample(X_train, y_train)

    # =========================
    # 2. Train
    # =========================
    model.fit(X_res, y_res)

    # =========================
    # 3. Predict
    # =========================
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    results = {
        "method": name,
        "model": model,
        "X_resampled": X_res,
        "y_resampled": y_res,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred)
    }

    print(f"\n=== {name} ===")
    print(f"Validation Accuracy: {results['val_accuracy']:.6f}")
    print(f"Test Accuracy: {results['test_accuracy']:.6f}")

    return results