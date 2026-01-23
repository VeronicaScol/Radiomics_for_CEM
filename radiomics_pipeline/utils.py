import sklearn
import numpy as np
import pandas as pd
import pickle
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


############ pre-process feature table ######################

def get_correlated_features_to_drop(thres_dataset_train: pd.DataFrame, corr_threshold: float = 0.85) -> np.ndarray:
    cor = thres_dataset_train.corr(method = 'spearman').abs()
    upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    to_drop = []
    for column in upper_tri.columns:
        for row in upper_tri.columns:
            val = upper_tri.at[row, column]
            if pd.notna(val) and val > corr_threshold:
                if np.nansum(upper_tri[column].values) > np.nansum(upper_tri[row].values):
                    to_drop.append(column)
                else:
                    to_drop.append(row)
    return np.unique(to_drop)


def preprocessing_train(df_train_features: pd.DataFrame,
                        variance_threshold: float = 0.01,
                        corr_threshold: float = 0.85):  ##patient name needs to be removed
    ##normalize the features
    df = df_train_features.copy()
    mean_std = {}
    for var in df.columns:
        temp_mean = df[var].mean()
        temp_std = df[var].std()
        mean_std[var] = (temp_mean, temp_std)
        # avoid division by zero
        df[var] = (df[var] - temp_mean) / (temp_std if temp_std != 0 else 1.0)

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df)
    thres_dataset_train = df.loc[:, selector.get_support()]

    to_drop = get_correlated_features_to_drop(thres_dataset_train, corr_threshold=corr_threshold)
    decor_dataset_train = thres_dataset_train.drop(to_drop, axis=1, errors="ignore")
    return mean_std, selector, to_drop, decor_dataset_train


def preprocessing_test(df_test_features: pd.DataFrame, mean_std, selector, to_drop) -> pd.DataFrame: ##apply parameters to test dataset
    df = df_test_features.copy()
    for var in df.columns:
        if var not in mean_std:
            # if columns mismatch, leave as-is; downstream selection will drop
            continue
        m, s = mean_std[var]
        df[var] = (df[var] - m) / (s if s != 0 else 1.0)

    thres_dataset_test = df.loc[:, selector.get_support()]
    decor_dataset_test = thres_dataset_test.drop(to_drop, axis=1, errors="ignore")
    return decor_dataset_test    

#####################Prediction helpers######################

def predict_proba_1(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise AttributeError("Model must implement predict_proba or decision_function.")


##################### generate results #################


def get_optimal_threshold(true_outcome, predictions):
    ##to obtain a good threshold based on the train dataset
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_outcome, predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def get_results(y_label, y_pred, label,
                optimal_threshold):  # better function below to get results with confidence interval
    ##optimal threshold: reuse the one computed on the train dataset
    ##label: index of the dataframe, can be "external radiomics results"
    ##returns a dataframe with auc accuracy precision recall f1-score
    dict_results = {}
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_label, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    dict_results["auc"] = roc_auc
    y_pred_binary = (np.array(y_pred) > optimal_threshold).astype(int)
    dict_results["accuracy"] = [sklearn.metrics.accuracy_score(y_label, y_pred_binary)]
    dict_results["precision"] = [sklearn.metrics.precision_score(y_label, y_pred_binary)]
    dict_results["recall"] = [sklearn.metrics.recall_score(y_label, y_pred_binary)]
    dict_results["f1 score"] = [sklearn.metrics.f1_score(y_label, y_pred_binary)]
    df_results = pd.DataFrame.from_dict(dict_results)
    df_results = df_results.reset_index(drop=True)
    df_results.index = [label]
    return df_results


np.random.seed(32)


def bootstrap(label, pred, f, nsamples=2000, random_state = 32):
    rng = np.random.default_rng(random_state)
    stats = []
    n = label.shape [0]
    for _ in range(nsamples):
        idx = rng.integers(0, n, size=n)
        stats.append(f(label[idx], pred[idx]))
    return stats, np.percentile(stats, (2.5, 97.5))


def nom_den(label, pred, f):
    if f == sklearn.metrics.accuracy_score:
        n = np.sum(label == pred)
        d = len(pred)
    if f == sklearn.metrics.precision_score:
        n = np.sum(pred[label == 1])
        d = np.sum(pred)
    if f == sklearn.metrics.recall_score:
        n = np.sum(pred[label == 1])
        d = np.sum(label)
    if f == sklearn.metrics.f1_score:
        n = 0
        d = 0
    return n, d


def get_ci(label, pred, f, nsamples=2000):
    stats, ci = bootstrap(label, pred, f, nsamples=nsamples)
    n, d = nom_den(label, pred, f)
    return stats, ["%5d/%5d (%5d %% )  CI [%0.2f,%0.2f]" %
                   (n, d, int(f(label, pred) * 100), ci[0], ci[1])]
  # doesn't compute the mean of the score


def get_ci_for_auc(label, pred, nsamples=2000, random_state = 32):
    rng = np.random.default_rng(random_state)    
    auc_values = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    n= label.shape[0]
    for _ in range(nsamples):
        idx = rng.integers(0, n, size=n)
        temp_pred = pred[idx]
        temp_fpr, temp_tpr, _ = sklearn.metrics.roc_curve(label[idx], temp_pred)
        roc_auc = sklearn.metrics.auc(temp_fpr, temp_tpr)
        auc_values.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, temp_fpr, temp_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ci_auc = np.percentile(auc_values, (2.5, 97.5))
    fpr, tpr, _ = sklearn.metrics.roc_curve(label, pred)
    return auc_values, ["%0.2f CI [%0.2f,%0.2f]" % (sklearn.metrics.auc(fpr, tpr), ci_auc[0], ci_auc[1])], [ci_auc[0], ci_auc[1]], mean_tpr


def get_stats_with_ci(y_label, y_pred, label, optimal_threshold, nsamples=2000):
    ##optimal threshold: reuse the one computed on the train dataset
    ##label: index of the dataframe, can be "external radiomics results"
    ##returns a dataframe with auc accuracy precision recall f1-score
    dict_results = {}
    dict_distributions = {}
    
    dict_distributions["auc"], dict_results["auc"], _, _ = get_ci_for_auc(y_label, y_pred, nsamples = nsamples)
    
    y_pred_binary = (np.array(y_pred) > optimal_threshold).astype(int)
    
    dict_distributions["accuracy"], dict_results["accuracy"] = get_ci(y_label, y_pred_binary, sklearn.metrics.accuracy_score, nsamples = nsamples)
    dict_distributions["precision"], dict_results["precision"] = get_ci(y_label, y_pred_binary, sklearn.metrics.precision_score, nsamples = nsamples)
    dict_distributions["specificity"], dict_results["specificity"] = get_ci(np.ones(len(y_label)) - y_label, np.ones(len(y_pred_binary)) - y_pred_binary, sklearn.metrics.recall_score, nsamples = nsamples)
    dict_distributions["recall"], dict_results["recall"] = get_ci(y_label, y_pred_binary, sklearn.metrics.recall_score, nsamples=nsamples)
    dict_distributions["f1 score"], dict_results["f1 score"] = get_ci(y_label, y_pred_binary, sklearn.metrics.f1_score, nsamples=nsamples)
    
    df_results = pd.DataFrame.from_dict(dict_results).reset_index(drop=True)
    df_results.index = [label]
    df_distributions = pd.DataFrame.from_dict(dict_distributions).reset_index(drop=True)
    return df_distributions, df_results


######################### save and load model and results ############################

def save_all_params(path_to_save, rfe, filtered_col, gsearch, mean_std, to_drop, selector, support, proba_train,
                    proba_test, proba_external, data_used="radiomics"):
    ##give the path to save
    filename_filtered_col = path_to_save + 'filtered_col_' + data_used + '.pkl'
    pickle.dump(rfe, open(filename_filtered_col, 'wb'))
    filename_rfe = path_to_save + 'rfe_' + data_used + '.pkl'
    pickle.dump(rfe, open(filename_rfe, 'wb'))
    filename_gsearch = path_to_save + 'gsearch_' + data_used + '.pkl'
    pickle.dump(gsearch, open(filename_gsearch, 'wb'))
    filename_parameters = path_to_save + r"parameters_" + data_used + ".pkl"
    pickle.dump([mean_std, selector, to_drop, support], open(filename_parameters, 'wb'))
    filename_proba_train = path_to_save + r"proba_train_" + data_used + ".pkl"
    pickle.dump(proba_train, open(filename_proba_train, 'wb'))
    filename_proba_test = path_to_save + r"proba_test_" + data_used + ".pkl"
    pickle.dump(proba_test, open(filename_proba_test, 'wb'))
    filename_proba_external = path_to_save + r"proba_external_" + data_used + ".pkl"
    pickle.dump(proba_external, open(filename_proba_external, 'wb'))
    return "done"


def load_all_params(path_to_load, data_used="radiomics"):
    ##give the path to load
    filename_filtered_col = path_to_load + 'filtered_col_' + data_used + '.pkl'
    filtered_col = pickle.load(open(filename_filtered_col, 'rb'))
    filename_rfe = path_to_load + 'rfe_' + data_used + '.pkl'
    rfe = pickle.load(open(filename_rfe, 'rb'))
    filename_gsearch = path_to_load + 'gsearch_' + data_used + '.pkl'
    gsearch = pickle.load(open(filename_gsearch, 'rb'))
    filename_parameters = path_to_load + r"parameters_" + data_used + ".pkl"
    [mean_std, selector, to_drop, support] = pickle.load(open(filename_parameters, 'rb'))
    filename_proba_train = path_to_load + r"proba_train_" + data_used + ".pkl"
    proba_train = pickle.load(open(filename_proba_train, 'rb'))
    filename_proba_test = path_to_load + r"proba_test_" + data_used + ".pkl"
    proba_test = pickle.load(open(filename_proba_test, 'rb'))
    filename_proba_external = path_to_load + r"proba_external_" + data_used + ".pkl"
    proba_external = pickle.load(open(filename_proba_external, 'rb'))
    return rfe, filtered_col, gsearch, mean_std, to_drop, selector, support, proba_train, proba_test, proba_external

################Persistence helpers######################
def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

#########################MAIN FUNC###########################3333


