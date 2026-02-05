import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None  # to be set after fitting

        if self.k <= 0:
            raise ValueError("k must be positive")
        if type(self.k) != int:
            raise TypeError("k must be an integer")
        
        if self.tol < 0:
            raise ValueError("tolerance must be non-negative")
        if type(self.tol) not in [float, int]:
            raise TypeError("tolerance must be a float or integer")
        
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if type(self.max_iter) != int:
            raise TypeError("max_iter must be an integer")

    def _initialize_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        Initializes k centroids randomly from the data points in mat.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        indices = np.random.choice(mat.shape[0], self.k, replace=False)
        centroids = mat[indices]
        return centroids

    def _fit_validation(self, mat: np.ndarray):
        """
        Validates the input matrix for fitting.

        """
        if mat.ndim != 2:
            raise ValueError("Input must be (n_samples, n_features)")

        n_features = mat.shape[1]

        if hasattr(self, "n_features_"):
            if n_features != self.n_features_:
                raise ValueError("Mismatched feature dimensions")
        else:
            self.n_features_ = n_features

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        self._fit_validation(mat)
        
        centroids = self._initialize_centroids(mat)

        for i in range(self.max_iter):
            dist = cdist(mat, centroids) # calculate distances from points to centroids
            labels = np.argmin(dist, axis=1) # assign labels based on closest centroid

            # create new centroids
            new_centroids = np.array([mat[labels == j].mean(axis=0) if np.sum(labels == j) > 0 
                else centroids[j] # keep old centroid
                for j in range(self.k)]) #loop thru

            diff = centroids - new_centroids
            shift = np.sqrt(np.sum(diff ** 2))  # calc Euclidean distance
            centroids = new_centroids # update

            if shift < self.tol: #convergence check
                break

        self.centroids = centroids # store for predict
        # for error calculation
        self.points = mat.copy()
        self.labels = labels.copy()


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        
        # dimension match
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError(f"Expected {self.centroids.shape[1]} features, got {mat.shape[1]}")
        
        dist = cdist(mat, self.centroids) # calculate distances from points to centroids
        labels = np.argmin(dist, axis=1) # assign labels based on closest centroid

        return labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")

        dist = cdist(self.points, self.centroids)
        assigned_dists = dist[np.arange(len(self.labels)), self.labels]
        return np.sum(assigned_dists ** 2) / len(self.points)  # Mean squared error


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' first.")
        return self.centroids