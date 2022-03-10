from random import shuffle


def random_split(X, y=None, frac=0.8):
    """Random split of data

    Parameters
    ----------
    X
        Inputs
    y, optional
        Outputs, by default None
    frac, optional
        Fraction of data to use as training data, by default 0.8
    """
    idx = list(range(len(X)))

    if y is not None:
        assert len(y) == len(X)

    shuffle(idx)
    n_train = int(frac * len(X))

    idx_train, idx_test = idx[:n_train], idx[n_train:]

    if y is not None:
        return X[idx_train], y[idx_train], X[idx_test], y[idx_test]
    else:
        return X[idx_train], X[idx_test]
