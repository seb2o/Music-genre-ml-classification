from typing import Union

import numpy as np
import pandas as pd
import seaborn as sn
import utils
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from Code import classifiers


def conf_matrix(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    Displays the confusion matrix of the model predictions
    :param y_pred: the model predictions
    :param y_true: the true labels
    """
    confMatrix = pd.DataFrame(confusion_matrix(y_true, y_pred), utils.genreNames, utils.genreNames)
    sn.set(font_scale=1.4)
    sn.heatmap(confMatrix, annot=True, annot_kws={"size": 16}, fmt='d')
    plt.show()


def distribution(pred: Union[np.ndarray, pd.DataFrame], true: Union[np.ndarray, pd.DataFrame]) -> None:
    """
    Displays the distribution of the predicted and true labels on top of each others
    :param pred:
    :param true:
    """
    plt.figure(figsize=(15, 8))
    plt.bar(utils.genreNames, np.unique(pred, return_counts=True)[1], color='tab:blue')
    plt.bar(utils.genreNames, np.unique(true, return_counts=True)[1], alpha=0.5, color='tab:blue')


def multiclass_performance_metrics(y_pred: np.ndarray, y_true: np.ndarray,
                                   labels: list[str] = utils.genreNames) -> pd.DataFrame:
    """
    Computes for each class as if it was binary classification the true/false positive/negative rates
    :param labels: if Specified, replace the target classes indices by the corresponding string in the list
    :param y_pred:
    :param y_true:
    :return: a data frame with the number of each catagorie for each class, and the f1 score for each class
    """
    results = pd.DataFrame(columns=["tp", "tn", "fp", "fn"])
    comp = pd.DataFrame({
        "y_pred": y_pred,
        "y_true": y_true
    })

    for current_class in range(10):
        current_tp = comp[(comp['y_pred'] == current_class) & (comp['y_true'] == current_class)]
        current_tn = comp[(comp['y_pred'] != current_class) & (comp['y_true'] != current_class)]
        current_fp = comp[(comp['y_pred'] == current_class) & (comp['y_true'] != current_class)]
        current_fn = comp[(comp['y_pred'] != current_class) & (comp['y_true'] == current_class)]
        results.loc[labels[current_class]] = [len(current_tp), len(current_tn), len(current_fp), len(current_fn)]

    results["F1Score"] = 2 * results.tp / (2 * results.tp + results.fn + results.fp)

    return results


def evaluate_ensemble_lstm(ensemble_y_pred, y_test):
    """
    Displays confusion matrix of the combined predictions, returns its performance along the accuracies of
    the individual models
    :param ensemble_y_pred: a dataframe with a column by model and a row by sample. The combined predictions
        should be named "combined"

    :param y_test: true label with, for each track, the same label for each sample
    :return: accuracies, performance
    """
    y_test_grouped = np.array([track_labels[0] for track_labels in y_test])
    true = pd.Series(y_test_grouped, name="true")

    scores = ensemble_y_pred.apply(lambda col: np.mean(col == true))
    scores["mean"] = scores.drop(columns="combined").mean()
    scores = scores.rename("accuracies").sort_values(ascending=False)

    y_pred_combined = ensemble_y_pred.combined.values
    perf = multiclass_performance_metrics(y_pred_combined, y_test_grouped)
    conf_matrix(y_pred_combined, y_test_grouped)
    return scores, perf
