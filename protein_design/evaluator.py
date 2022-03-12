from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import ndcg_score
import plotly.express as px


def regression_metrics(y_true, y_pred, plot=False, hover_data=None) -> Dict[str, float]:
    """Compute common metrics used to evalute regression models

    Parameters
    ----------
    y_true
        Actual values
    y_pred
        Predicted values
    plot
        If True, plots a scatterplot of true vs. predicted values, by default False
    hover_data
        If provided, includes this metadata as plotly hover_data

    Returns
    -------
        Dictionary mapping regression metric to value
    """
    spearman = spearmanr(y_true, y_pred)[0]
    pearson = pearsonr(y_true, y_pred)[0].item()
    mae = np.mean(np.abs(y_true - y_pred))
    idx = np.argsort(-y_true)
    rank = np.argsort(idx)
    ndcg = ndcg_score(rank.reshape(1, -1), -y_pred.reshape(1, -1))

    if plot:
        title = f"Spearman: {spearman:.3f} Pearson: {pearson:.3f} MAE: {mae:.3f} NDCG: {ndcg:.3f}"
        df = pd.DataFrame()
        df["y_true"] = y_true
        df["y_pred"] = y_pred
        df["hover"] = hover_data
        fig = px.scatter(df, x="y_true", y="y_pred", title=title, hover_data=["hover"])
        fig.show()

    return {"spearman": spearman, "pearson": pearson, "mae": mae, "ndcg": ndcg}
