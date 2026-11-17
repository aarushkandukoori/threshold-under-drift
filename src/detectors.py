"""Two frozen detectors. They are trained once on clean normal data and never updated.
Higher score = more anomalous."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


class FrozenIForest:
    name = "iforest"

    def __init__(self, seed):
        self.model = IsolationForest(n_estimators=200, random_state=seed)

    def fit(self, X):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class FrozenKNN:
    """Mean distance to the 10 nearest training rows."""
    name = "knn"

    def __init__(self, seed=None, k=10):
        self.k = k

    def fit(self, X):
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def score(self, X):
        dist, _ = self.nn.kneighbors(X)
        return dist.mean(axis=1)


DETECTORS = {"iforest": FrozenIForest, "knn": FrozenKNN}
