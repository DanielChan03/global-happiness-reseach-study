
import pandas as pd
import numpy as np
import inspect
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, matthews_corrcoef

# model training function
def train_ensemble_models(X_train, y_train, X_val, y_val, X_test, y_test, models, sort_by=None):
    """
    Train, tune, and evaluate ensemble models on preprocessed data.

    Parameters:
    -----------
    X_train, X_val, X_test : preprocessed feature arrays
    y_train, y_val, y_test : target arrays/Series
    models : list of dicts
        Each dict should contain:
            - 'name': str, model name
            - 'estimator': sklearn-compatible classifier
            - 'param_grid'(optional): dict of hyperparameters for GridSearchCV
    sort_by : sort the results by Validation Accuracy by default.
        Modifiable:
            "None" by default, 
            "Validation Accuracy",
            "Test Accuracy",
            "Validation MCC",
            "Test MCC"
            
    Returns:
    --------
    results_df : pd.DataFrame
        Sorted results of models by Test Accuracy
    best_models : dict
        Best model per algorithm keyed by model name
    """

    # Helper function to ensure y is 1-D
    def flatten_y(y):
        # Get the variable name from the caller's local variables
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        var_names = [name for name, val in callers_local_vars if val is y]
        var_name = var_names[0] if var_names else "y"

        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                print(f"{var_name}: Converted to 1D pd automatically...")
                return y.iloc[:, 0]
            else:
                raise ValueError(f"{var_name} has multiple columns ({y.shape[1]}), cannot flatten automatically")
        elif isinstance(y, pd.Series):
            print(f"{var_name}: Already 1D pd.Series")
            return y
        elif isinstance(y, (np.ndarray, list)):
            print(f"{var_name}: Converted to 1D np automatically...")
            return np.array(y).ravel()
        else:
            raise TypeError(f"{var_name}: Unsupported y type: {type(y)}")

    # Flatten targets
    y_train = flatten_y(y_train)
    y_test  = flatten_y(y_test)
    y_val   = flatten_y(y_val)

    results = []
    best_models = {}

    for m in models:
        print(f"\nTraining {m['name']}...")
        if "param_grid" in m and m["param_grid"] is not None:
            grid = GridSearchCV(
                estimator=m['estimator'],
                param_grid=m['param_grid'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            best_params = grid.best_params_

        else:
            # No tuning → pure baseline
            model = m["estimator"]
            model.fit(X_train, y_train)
            best_params = None

        # Evaluate
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)

        y_val_mcc = matthews_corrcoef(y_val, y_val_pred)
        y_test_mcc = matthews_corrcoef(y_test, y_test_pred)

        results.append({
            "Model": m["name"],
            "Best Params": best_params,
            "Validation Accuracy": val_acc,
            "Test Accuracy": test_acc,
            "Validation MCC": y_val_mcc,
            "Test MCC": y_test_mcc
        })

        best_models[m["name"]] = {
            "model": model,
            "test_accuracy": test_acc
        }

    results_df = pd.DataFrame(results)

    if sort_by is not None:
        results_df = results_df.sort_values(by=sort_by, ascending=False)

    return results_df, best_models
