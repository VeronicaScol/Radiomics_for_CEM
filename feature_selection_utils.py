# feature_selection.py
##Quick “when to use what”
    #1. Filter: fast baseline, good first pass, works well when you have many features.
        ## filter_method_cv
    #2. Wrapper (RFE): can be strong but expensive; good when feature count isn’t huge.
        ## 2a. rfe_no_cv
        ## 2b. rfe_with_cv
    #3. Embedded: good tradeoff; often the default practical choice.
        ## 3a. embedded_method (Random forest based)
        ## 3b. embedded_l1_logregcv (lasso)
   

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone

#1.-----------------------------------------------------------------------------------------------
def filter_method_cv(
    X,
    y,
    model,
    k_values=(5, 10, 15, 20, 30, 40, 45, 50, 60, 70, 80, 90, 100, "all"),
    score_func=f_classif,
    scoring="roc_auc",
    n_splits=10,
    shuffle=True,
    random_state=27,
    n_jobs=-1,
    refit=True,
    return_search=False,
    return_support=False,
):
    """
    Filter-based feature selection where k is chosen via CV performance
    using a Pipeline(SelectKBest -> model) and GridSearchCV.

    Returns
    -------
    selected_features : list
        Names (if DataFrame) or indices (if array) of selected features using best k.
    best_k : int or "all"
        Best k found by CV.
    best_score : float
        Best mean CV score.
    (optional) gsearch : GridSearchCV
        Returned only if return_search=True
    """
    # Build pipeline
    pipe = Pipeline(
        steps=[
            ("anova", SelectKBest(score_func=score_func)),
            ("model", clone(model)),
        ]
    )

    # Grid of k values
    param_grid = {"anova__k": list(k_values)}

    # CV strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Grid search
    gsearch = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=refit,
    )

    # Fit
    gsearch.fit(X, y)

    best_k = gsearch.best_params_["anova__k"]
    best_score = gsearch.best_score_

    # Pull fitted selector from best estimator (only valid if refit=True)
    best_selector = gsearch.best_estimator_.named_steps["anova"]
    support_mask = best_selector.get_support()

    # Convert support mask to feature names/indices
    if isinstance(X, pd.DataFrame):
        selected_features = list(X.columns[support_mask])
    else:
        selected_features = list(np.where(support_mask)[0])

    if return_search:
        if return_support:
            return selected_features, best_k, best_score, support_mask, gsearch
        return selected_features, best_k, best_score, gsearch

    if return_support:
        return selected_features, best_k, best_score, support_mask
    return selected_features, best_k, best_score


#2.-----------------------------------------------------------------------------------------------
#2a.---------------------------------------------------------------------------------------------
def rfe_no_cv(X, y, n_features=10, estimator=None, return_support=False):
    """
    Wrapper-based feature selection using Recursive Feature Elimination WITHOUT cross validation.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable = Outcome
    n_features : int, default=10
        Number of features to select
    estimator : estimator object, default=None
        A supervised learning estimator with a fit method
        
    Returns:
    --------
    selected_features : list
        Names of selected features
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=27)
    
    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    selector.fit(X, y)

    support_mask = selector.get_support()
    
    # Get feature names if X is a DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        selected_features = feature_names[selector.get_support()]
        
        # Optional: Plot feature ranking
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, selector.ranking_)
        plt.xticks(rotation=90)
        plt.title('Feature Ranking (lower is better)')
        plt.tight_layout()
        plt.show()
        
        if return_support:
            return list(selected_features), support_mask
        return list(selected_features)
    else:
        selected_idx = list(np.where(support_mask)[0])
        if return_support:
            return selected_idx, support_mask
        return selected_idx

#2b. ---------------------------------------------------------------------------------------------
def rfe_with_cv(
    X,
    y,
    estimator=None,
    step=1,
    min_features_to_select=10,
    n_splits=10,
    shuffle=True,
    random_state=27,
    scoring="roc_auc",
    return_support=False,
):
    """
    Wrapper-based feature selection using RFECV (RFE + cross-validation).

    Returns
    -------
    selected_features : list
        Names of selected features (DataFrame) or indices (array)
    best_n_features : int
        Number of features chosen by CV
    best_score : float or None
        Best mean CV score if available (depends on sklearn version)
    selector : RFECV
        The fitted RFECV object (for inspection)
    """
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=27)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,  # remove if your sklearn version doesn't support it
    )
    selector.fit(X, y)

    # Selected mask
    support = selector.support_
    support_mask = support
    best_n_features = selector.n_features_

    # Some sklearn versions expose best score via cv_results_ (more reliable)
    best_score = None
    if hasattr(selector, "cv_results_") and "mean_test_score" in selector.cv_results_:
        best_score = float(np.max(selector.cv_results_["mean_test_score"]))

    if isinstance(X, pd.DataFrame):
        selected_features = list(X.columns[support])
    else:
        selected_features = list(np.where(support)[0])

    if return_support:
        return selected_features, best_n_features, best_score, support_mask, selector
    return selected_features, best_n_features, best_score, selector


#3. ----------------------------------------------------------------------------------------------
#3a. -------------------------------------------------------------------------------------------
def embedded_method(X, y, n_features=10, model=None, threshold=None, return_support=False):
    """
    Embedded feature selection using model's feature importance.
    
    Parameters:
    -----------
    X : DataFrame or array
        Feature matrix
    y : Series or array
        Target variable
    n_features : int, default=10
        Number of features to select (used if threshold is None)
    model : estimator object, default=None
        A supervised learning estimator with feature_importances_ or coef_ attribute
    threshold : float, default=None
        Threshold value for feature selection
        
    Returns:
    --------
    selected_features : list
        Names of selected features
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=27)
    
    if threshold is not None:
        selector = SelectFromModel(model, threshold=threshold)
    else:
        selector = SelectFromModel(model, max_features=n_features)
    
    selector.fit(X, y)
    support_mask = selector.get_support()
    
    # Get feature names if X is a DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        selected_features = feature_names[support_mask]
        
        # Plot feature importances if available
        if hasattr(selector.estimator_, 'feature_importances_'):
            importances = selector.estimator_.feature_importances_
        elif hasattr(selector.estimator_, 'coef_'):
            importances = np.abs(selector.estimator_.coef_).flatten()
        else:
            importances = None
            
        if importances is not None:
            plt.figure(figsize=(12, 6))
            plt.bar(feature_names, importances)
            plt.xticks(rotation=90)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
        
        if return_support:
            return list(selected_features), support_mask
        return list(selected_features)
    else:
        selected_idx = list(np.where(support_mask)[0])
        if return_support:
            return selected_idx, support_mask
        return selected_idx

#3b. ---------------------------------------------------------------------------------------------
def embedded_l1_logregcv(
    X,
    y,
    cv=10,
    scoring="roc_auc",
    solver="saga",
    max_iter=10000,
    n_jobs=-1,
    random_state=27,
    Cs=10,
    fit_intercept=True,
    class_weight=None,
    tol=1e-4,
    refit=True,
    return_support=False,
):
    """
    Embedded feature selection via L1-penalized LogisticRegressionCV.
    Uses scaling + L1 logistic regression; selected features are those with non-zero coefficients.

    Returns
    -------
    selected_features : list
        Selected feature names (DataFrame) or indices (array)
    best_C : float
        Best inverse regularization strength chosen by CV
    n_selected : int
        Number of selected features
    fitted_pipe : Pipeline
        Fitted pipeline: StandardScaler -> LogisticRegressionCV
    """

    # Build model with your parameters (kept as much as possible)
    logit_l1 = LogisticRegressionCV(
        penalty="l1",
        solver=solver,
        scoring=scoring,
        cv=cv,
        max_iter=max_iter,
        n_jobs=n_jobs,
        random_state=random_state,
        Cs=Cs,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
        tol=tol,
        refit=refit,
    )

    # Scale then fit model (important for L1 logistic)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", logit_l1),
    ])

    pipe.fit(X, y)

    # Extract coefficients from fitted LogisticRegressionCV
    coef = pipe.named_steps["model"].coef_.ravel()
    support_mask = coef != 0

    # Best C: LogisticRegressionCV stores chosen C_ as array-like
    best_C = float(pipe.named_steps["model"].C_[0])

    n_selected = int(support_mask.sum())

    if isinstance(X, pd.DataFrame):
        selected_features = list(X.columns[support_mask])
    else:
        selected_features = list(np.where(support_mask)[0])

    if return_support:
        return selected_features, best_C, n_selected, support_mask, pipe
    return selected_features, best_C, n_selected, pipe

    
