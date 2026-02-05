# write your silhouette score unit tests here
import pytest
import numpy as np
from cluster.silhouette import Silhouette
from cluster.utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)
from sklearn.metrics import silhouette_samples, silhouette_score

def test_match_sklearn():
    """

    Test that silhouette scores match sklearn's implementation.

    """

    X, y = make_clusters(scale=0.5, k=3)

    sil = Silhouette()
    my_scores = sil.scores(X, y)
    sk_scores = silhouette_samples(X, y)

    assert my_scores.shape == sk_scores.shape
    assert np.allclose(my_scores, sk_scores, atol=1e-4)


def test_tight_clusters():
    """

    Test that tight clusters have higher silhouette scores than loose clusters.

    """

    X_tight, y_tight = make_clusters(scale=0.3)
    X_loose, y_loose = make_clusters(scale=2.0)

    sil = Silhouette()
    tight_score = sil.score(X_tight, y_tight)
    loose_score = sil.score(X_loose, y_loose)

    assert tight_score > loose_score


def test_high_dimensional():
    """

    Test that high-dimensional data works.

    """

    X, y = make_clusters(n=500, m=200, k=3)

    scores = Silhouette().scores(X, y)

    assert scores.shape == (X.shape[0],)
    assert np.all(scores >= -1)
    assert np.all(scores <= 1)


def test_single_cluster_raises_error():
    """

    Test that 1 cluster raises a ValueError.

    """

    X, _ = make_clusters()
    labels = np.zeros(X.shape[0])

    with pytest.raises(ValueError):
        Silhouette().score(X, labels)


def test_mismatched_shapes_raise_error():
    """

    Test for mismatch between data and labels.

    """

    X, y = make_clusters()
    with pytest.raises(ValueError):
        Silhouette().score(X[:-1], y)
