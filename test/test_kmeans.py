# Write your k-means unit tests here
import pytest
import numpy as np
from cluster.kmeans import KMeans
from cluster.utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

@pytest.fixture
def test_datasets():
    """
    
    Datasets for testing KMeans implementation.
    
    """
    t_clusters, t_labels = make_clusters(scale=0.3)
    l_clusters, l_labels = make_clusters(scale=2)
    m_clusters, m_labels = make_clusters(k=10)
    d_clusters, d_labels = make_clusters(n=1000, m=200, k=3)
    
    return {
        "tight": (t_clusters, t_labels),
        "loose": (l_clusters, l_labels),
        "many_clusters": (m_clusters, m_labels),
        "high_dim": (d_clusters, d_labels)
        }


def test_initialization():
    """
    
    Test initialization with valid and invalid inputs.
    
    """

    # Valid initialization
    kmeans = KMeans(k=3, tol=1e-4, max_iter=50)
    assert kmeans.k == 3
    assert kmeans.tol == 1e-4
    assert kmeans.max_iter == 50

    # Invalid k values
    with pytest.raises(ValueError):
        KMeans(k=-1)  # k must be positive
    with pytest.raises(TypeError):
        KMeans(k="3")  # k must be an integer

    # Invalid tolerance values
    with pytest.raises(ValueError):
        KMeans(k=3, tol=-1)  # tol must be non-negative
    with pytest.raises(TypeError):
        KMeans(k=3, tol="0.01")  # tol must be a float or integer

    # Invalid max_iter values
    with pytest.raises(ValueError):
        KMeans(k=3, max_iter=0)  # max_iter must be positive
    with pytest.raises(TypeError):
        KMeans(k=3, max_iter="50")  # max_iter must be an integer


def test_fit(test_datasets):
    """
    
    Test fitting on clustered data.
    
    """
    t_clusters, _ = test_datasets["tight"]
    kmeans = KMeans(k=3)
    kmeans.fit(t_clusters)
    centroids = kmeans.get_centroids()

    # Assert centroids are not None and have the correct shape
    assert centroids is not None
    assert centroids.shape == (3, t_clusters.shape[1])

    # Test fitting on data with mismatched dimensions
    with pytest.raises(ValueError):
        kmeans.fit(np.random.rand(10, 5))  # Should fail if dimensions don't match


def test_predict(test_datasets):
    """
    
    Test predictions.
    
    """
    t_clusters, _ = test_datasets["tight"]
    kmeans = KMeans(k=3)
    kmeans.fit(t_clusters)
    pred_labels = kmeans.predict(t_clusters)

    # Assert predicted labels are the correct size
    assert pred_labels.shape == (t_clusters.shape[0],)

    # Test predictions on data with mismatched dimensions
    with pytest.raises(ValueError):
        kmeans.predict(np.random.rand(10, 5))  # Should fail if dimensions don't match


def test_get_error(test_datasets):
    """
    
    Test error calculation after fitting.
    
    """
    t_clusters, _ = test_datasets["tight"]
    kmeans = KMeans(k=3)
    kmeans.fit(t_clusters)
    error = kmeans.get_error()

    # Assert error is a float and non-negative
    assert isinstance(error, float)
    assert error >= 0

    # Test error calculation without fitting the model
    kmeans_unfit = KMeans(k=3)
    with pytest.raises(ValueError):
        kmeans_unfit.get_error()  # Should fail if model hasn't been fit yet


def test_get_centroids(test_datasets):
    """
    
    Test get centroids after fitting.
    
    """
    t_clusters, _ = test_datasets["tight"]
    kmeans = KMeans(k=3)
    kmeans.fit(t_clusters)
    centroids = kmeans.get_centroids()

    # Assert centroids are not None and have the correct shape
    assert centroids is not None
    assert centroids.shape == (3, t_clusters.shape[1])

    # Test centroid retrieval without fitting the model
    kmeans_unfit = KMeans(k=3)
    with pytest.raises(ValueError):
        kmeans_unfit.get_centroids()  # Should fail if model hasn't been fit yet