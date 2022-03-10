from typing import Dict

import numpy as np
from scipy.stats import spearmanr, pearsonr
import plotly.express as px


def regression_metrics(y_true, y_pred, plot=False) -> Dict[str, float]:
    """Compute common metrics used to evalute regression models

    Parameters
    ----------
    y_true
        Actual values
    y_pred
        Predicted values
    plot
        If True, plots a scatterplot of true vs. predicted values, by default False

    Returns
    -------
        Dictionary mapping regression metric to value
    """
    spearman = spearmanr(y_true, y_pred)[0]
    pearson = pearsonr(y_true, y_pred)[0]
    mae = np.mean(np.abs(y_true - y_pred))

    if plot:
        fig = px.scatter(x=y_true, y=y_pred)
        fig.show()

    return {'spearman': spearman, 'pearson': pearson, 'mae': mae}
