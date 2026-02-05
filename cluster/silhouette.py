import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """

        inputs:
            none

        """

    def _sil_validation(self, X: np.ndarray, y: np.ndarray):
        """

        validates inputs for silhouette score calculation

        """

        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")

        if len(np.unique(y)) < 2:
            raise ValueError("Silhouette undefined for < 2 clusters")

    def scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """

        calculates the silhouette score for each of the observations
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`

        """
        self._sil_validation(X, y)
        dist_matrix = cdist(X, X) # pairwise distance matrix

        unique_labels = np.unique(y) # Get unique cluster labels, why?
        n = len(X)
        sil_scores = np.zeros(n)

        for i in range(n):
            label = y[i]

            same_cluster = (y == label) # points in the same cluster
            same_distances = dist_matrix[i, same_cluster]
            a = np.mean(same_distances[same_distances != 0])  # a(i), exclude self
        
            b1 = []
            for other_label in unique_labels:
                if other_label != label:
                    other_mask = (y == other_label)
                    b = np.mean(dist_matrix[i, other_mask])
                    b1.append(b)
            b = min(b1) # b(i)
        
            sil_scores[i] = (b - a) / max(a, b)
        return sil_scores

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
    
        calculates the average silhouette score

        """

        return np.mean(self.scores(X, y))