from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import roc_curve, roc_auc_score

from ethos.constants import PROJECT_ROOT, ADMISSION_STOKEN
from ethos.tokenize import SpecialToken

sns.set()


def compute_basic_metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "prevalence": y_true.mean(),
        "auc": roc_auc_score(y_true, y_pred),
        "auprc": -np.trapz(*roc_curve(y_true, y_pred)[:2]),
    }


def objective_function(std, points, equal_variance=False):
    thresholds = np.linspace(-10, 11, num=10000)
    std2 = std[0] if equal_variance else std[1]
    cdf_hypothesis_1 = norm.cdf(thresholds, loc=0, scale=std[0])
    cdf_hypothesis_2 = norm.cdf(thresholds, loc=1, scale=std2)
    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr_values = 1 - cdf_hypothesis_2
    fpr_values = 1 - cdf_hypothesis_1
    val = sum(np.min((tpr_values - tpr) ** 2 + (fpr_values - fpr) ** 2) for tpr, fpr in points)
    return val


def compute_gaussian_metrics(y_true, y_pred, equal_variance=False):
    fpr_points, tpr_points, _ = roc_curve(y_true, y_pred)
    points = np.stack((tpr_points, fpr_points), axis=1).tolist()

    # ==========================================================================
    # here starts the fitting of parametric ROC curve to experimental points
    # ==========================================================================
    delta = 1e-6
    upper_const = 10
    # Define the range constraints for x and y
    std_constraint = {
        "type": "ineq",
        "fun": lambda x: np.array([x[0] - delta, upper_const - x[0]]),
    }
    std2_constraint = {
        "type": "ineq",
        "fun": lambda x: np.array([x[1] - delta, upper_const - x[1]]),
    }
    if equal_variance:
        constraints = [std_constraint]
        x0 = np.array([1.0])
    else:
        constraints = [std_constraint, std2_constraint]
        x0 = np.array([0.5, 0.5])

    result = minimize(objective_function, x0, args=(points,), constraints=constraints)

    if equal_variance:
        optimal_x = result.x
    else:
        optimal_x, optimal_y = result.x

    # Parameters for hypothesis 1 (mean and standard deviation)
    mean_hypothesis_1 = 0.0
    std_dev_hypothesis_1 = optimal_x

    # Parameters for hypothesis 2 (mean and standard deviation)
    mean_hypothesis_2 = 1
    std_dev_hypothesis_2 = optimal_x
    if not equal_variance:
        std_dev_hypothesis_2 = optimal_y

    # Calculate the cumulative distribution functions (CDFs) for the two distributions
    thresholds = np.linspace(-5, 10, num=1000)
    cdf_hypothesis_1 = norm.cdf(thresholds, loc=mean_hypothesis_1, scale=std_dev_hypothesis_1)
    cdf_hypothesis_2 = norm.cdf(thresholds, loc=mean_hypothesis_2, scale=std_dev_hypothesis_2)

    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr_values = 1 - cdf_hypothesis_2
    fpr_values = 1 - cdf_hypothesis_1
    # Find the best operating point defined as the closest point to (0,1)
    dist_squared = fpr_values ** 2 + (tpr_values - 1) ** 2
    min_idx = np.argmin(dist_squared)
    fpr = fpr_values[min_idx]
    tpr = tpr_values[min_idx]

    # =======================================================
    # Compute metrics now using the operating point
    # =======================================================
    positives = (y_true == 1).sum()
    negatives = (y_true == 0).sum()

    tp = tpr * positives
    fn = (1 - tpr) * positives
    tn = (1 - fpr) * negatives
    fp = fpr * negatives

    tp_values = tpr_values * positives
    fp_values = fpr_values * negatives
    fn_values = (1 - tpr_values) * positives

    denominator = tp_values + fp_values
    not_zero = denominator != 0
    precision_values = np.divide(tp_values, denominator, where=not_zero)
    precision_values[~not_zero] = 1
    recall_values = tp_values / (tp_values + fn_values)

    # tier 1
    auprc = -np.trapz(precision_values, recall_values)
    auc = -np.trapz(tpr_values, fpr_values)
    accuracy = (tp + tn) / (positives + negatives)
    sensitivity = tpr
    specificity = 1 - fpr
    #    sens_plus_spec = sensitivity + specificity
    # tier 2
    # sensitivity
    # specificity
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    plr = sensitivity / (1 - specificity)
    nlr = (1 - sensitivity) / specificity
    f1 = tp / (tp + (fp + fn) / 2)

    return {
        # Tier 1
        "auc": auc,
        "auprc": auprc,
        "accuracy": accuracy,
        "sensitivity+specificity": sensitivity + specificity,
        # Tier 2
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "plr": plr,
        "nlr": nlr,
        "f1": f1,
        # Not needed for the sake of the challenge
        "recall": tpr,
        "precision_values": precision_values,
        "recall_values": recall_values,
        "tpr_points": tpr_points,
        "fpr_points": fpr_points,
        "tpr_values": tpr_values[::-1],
        "fpr_values": fpr_values[::-1],
    }


def print_auc_roc_plot(res, gaussian_res, title="AUC-ROC", lw=2, clinical=False):
    plt.plot([0, 1], [0, 1], color="grey", lw=lw, linestyle="--", label="Random Guess")
    plt.plot(
        gaussian_res["fpr_values"],
        gaussian_res["tpr_values"],
        color="darkorange",
        lw=lw,
        label="AUC-ROC Gaussian",
    )
    plt.scatter(gaussian_res["fpr_points"], gaussian_res["tpr_points"])
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)

    text = [
        f"N={res['n']:,}",
        f"prevalence={res['prevalence']:.2%}",
    ]
    if clinical:
        text.extend(
            [
                f"auc={gaussian_res['auc']:.3f}",
            ]
        )
    else:
        text.extend(
            [
                f"auc={res['auc']:.3f}",
                f"gauss_auc={gaussian_res['auc']:.3f}",
                f"gauss_f1-score={gaussian_res['f1']:.3f}",
                f"gauss_precision={gaussian_res['precision']:.3f}",
                f"gauss_recall={gaussian_res['recall']:.3f}",
                f"gauss_accuracy={gaussian_res['accuracy']:.3f}",
            ]
        )
    anc = AnchoredText("\n".join(text), loc="lower right", frameon=True, pad=0.3)
    anc.patch.set_boxstyle("round,pad=0.2")
    plt.gca().add_artist(anc)


def process_readmission_results(
        filename: str, admission_stoken: str, readmission_period: float
) -> pd.DataFrame:
    res_dir = PROJECT_ROOT / "results" / filename
    df = pd.concat(pd.read_json(res_path) for res_path in res_dir.iterdir())
    df.rename(columns={"actual": "actual_token", "patient_id": "subject_id"}, inplace=True)
    df["actual"] = (df.actual_token == admission_stoken).astype(int)
    df["expected"] = ((df.expected == 1) & (df.true_token_time <= readmission_period)).astype(int)
    discharge_idx_name = (
        "discharge_token_idx" if admission_stoken == ADMISSION_STOKEN else "discharge_idx"
    )
    df_gb = df.groupby(discharge_idx_name, dropna=False)
    return (
        df_gb.agg(
            {
                "subject_id": "first",
                "expected": "first",
                "actual": "mean",
                "true_token_time": "first",
                "true_token_dist": "first",
                "token_time": "mean",
                "token_dist": "mean",
                "patient_age": "first",
                discharge_idx_name: "first",
            }
        )
        .join(df_gb.agg(count=("actual", "count")))
        .reset_index(drop=True)
        .set_index("subject_id")
    )


def process_admission_results(filename: str, discharge_stoken: str) -> pd.DataFrame:
    res_dir = PROJECT_ROOT / "results" / filename
    df = pd.concat(pd.read_json(res_path) for res_path in res_dir.iterdir())
    df.rename(
        columns={
            "actual": "actual_token",
            "expected": "expected_token",
            "patient_id": "subject_id",
        },
        inplace=True,
    )
    prev_len = len(df)
    df = df.loc[df.actual_token.isin([discharge_stoken, SpecialToken.DEATH])]
    print(
        "Dropped rows due to an ambiguous result: {:,}/{:,} ({:.3%})".format(
            prev_len - len(df), prev_len, (prev_len - len(df)) / prev_len
        )
    )
    df["actual"] = (df.actual_token == SpecialToken.DEATH).astype(int)
    df["expected"] = (df.expected_token == SpecialToken.DEATH).astype(int)
    df_gb = df.groupby("admission_token_idx", dropna=False)
    agg_scheme = {
        "subject_id": "first",
        "expected": "first",
        "actual": "mean",
        "true_token_time": "first",
        "true_token_dist": "first",
        "token_time": "mean",
        "token_dist": "mean",
        "patient_age": "first",
    }
    if "stay_id" in df.columns:
        agg_scheme["stay_id"] = "first"

    def ci(df_: pd.DataFrame, sigmas: int = 2) -> bool:
        mean, std = df_.token_time.aggregate(["mean", "std"])
        true_time = df_.true_token_time.iloc[0]
        return abs(mean - true_time) < std * sigmas

    return (
        df_gb.agg(agg_scheme)
        .join(df_gb.agg(count=("actual", "count"), token_time_std=("token_time", "std")))
        .join(df_gb.apply(ci).rename("ci_2sig"))
        .join(df_gb.apply(partial(ci, sigmas=1)).rename("ci_1sig"))
        .reset_index(drop=True)
        .set_index("subject_id")
    )
