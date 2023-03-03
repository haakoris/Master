import pandas as pd
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from matplotlib import pyplot as plt
import numpy as np

rng = np.random.RandomState(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

def plot_cv_indices(cv, X, ax, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    n_samples = len(X)
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
    # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
            )
            
    # Formatting
    yticklabels = list(range(cv.n_splits))
    ax.set(
        yticks=np.arange(cv.n_splits) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[cv.n_splits + 0.2, -0.2],
        xlim=[0,n_samples + n_samples * 0.05],
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax


class TimeSeriesSplitSliding(TimeSeriesSplit):
    """Time Series cross-validator slightly modified from:
    https://ntguardian.wordpress.com/2017/06/19/walk-forward-analysis-demonstration-backtrader/ 
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    """

    def __init__(self, gap_size=0, train_splits=1, test_splits=1, fixed_length=False, **kwargs):
        super().__init__(**kwargs)
        self.gap_size = gap_size #TODO
        self.train_splits = train_splits
        self.test_splits = test_splits
        self.fixed_length = fixed_length

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        train_splits = self.train_splits
        test_splits = self.test_splits
        fixed_length = self.fixed_length
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        train_splits, test_splits = int(train_splits), int(test_splits)
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        if (n_folds - train_splits - test_splits) == 0 and (test_splits > 0):
            raise ValueError(
                ("Both train_splits and test_splits must be positive"
                 " integers."))
        indices = np.arange(n_samples)
        split_size = (n_samples // (n_folds + (train_splits + test_splits) - 2))
        test_size = split_size * test_splits
        train_size = split_size * train_splits
        test_starts = range(train_size + n_samples % n_folds,
                            n_samples - (test_size - split_size),
                            split_size)
        if fixed_length:
            for i, test_start in zip(range(len(test_starts)),
                                     test_starts):
                rem = 0
                if i == 0:
                    rem = n_samples % n_folds
                yield (indices[(test_start - train_size - rem):test_start],
                       indices[test_start:test_start + test_size])
        else:
            for test_start in test_starts:
                yield (indices[:test_start], indices[test_start:test_start + test_size])

if __name__ == "__main__":
    path = "/Users/eliassovikgunnarsson/Downloads/vix-daily_csv.csv"
    df = pd.read_csv(path, header = 0, index_col= 0)
    tscv = TimeSeriesSplitSliding(n_splits=5, train_splits=2, test_splits = 1, fixed_length=True)
    fig, ax = plt.subplots()
    X = df['VIX Close']
    plot_cv_indices(tscv, X, ax)
    plt.show()