from typing import Union

import numpy as np
import pandas as pd
import seaborn as sn
import utils
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def conf_matrix(y_pred: Union[np.ndarray, pd.DataFrame], y_true: Union[np.ndarray, pd.DataFrame]) -> None:
    """
    Displays the confusion matrix of the model predictions
    :param y_pred: the model predictions
    :param y_true: the true labels
    """
    confMatrix = pd.DataFrame(confusion_matrix(y_true, y_pred), utils.genreNames, utils.genreNames)
    cmap = sn.diverging_palette(220, 50, s=75, l=40, as_cmap=True)
    sn.set(font_scale=1.4)
    sn.heatmap(confMatrix, annot=True, annot_kws={"size": 16}, fmt='d', cmap=cmap)


def distribution(pred: Union[np.ndarray, pd.DataFrame], true: Union[np.ndarray, pd.DataFrame]) -> None:
    """
    Displays the distribution of the predicted and true labels on top of each others
    :param pred:
    :param true:
    """
    plt.figure(figsize=(15, 8))
    plt.bar(utils.genreNames, np.unique(pred, return_counts=True)[1], color='tab:blue')
    plt.bar(utils.genreNames, np.unique(true, return_counts=True)[1], alpha=0.5, color='tab:blue')
