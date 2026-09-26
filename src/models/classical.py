# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


# build ppns class to be suitable for pipeline& gridsearch
class PNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)

        # Lưu index theo từng class
        self.class_indices_ = {
            c: np.where(y == c)[0]
            for c in self.classes_
        }

        return self

    def _gaussian_kernel_matrix(self, X):

        """
        Tính Gaussian kernel giữa:
        X_test:  [n_test, n_features]
        X_train: [n_train, n_features]

        Output:
        [n_test, n_train]
        """

        # ||x-y||²
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)

        train_sq = np.sum(
            self.X_train_ ** 2,
            axis=1,
            keepdims=True
        ).T

        distances_sq = (
            X_sq
            + train_sq
            - 2 * X @ self.X_train_.T
        )

        distances_sq = np.maximum(
            distances_sq,
            0
        )

        sigma = max(
            float(self.sigma),
            1e-5
        )

        return np.exp(
            -distances_sq /
            (2 * sigma ** 2)
        )

    def predict(self, X):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        kernel = self._gaussian_kernel_matrix(X)

        scores = []

        for c in self.classes_:

            idx = self.class_indices_[c]

            # P(class | x)
            class_score = np.mean(
                kernel[:, idx],
                axis=1
            )

            scores.append(class_score)

        scores = np.column_stack(scores)

        return self.classes_[
            np.argmax(scores, axis=1)
        ]

    def predict_proba(self, X):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        kernel = self._gaussian_kernel_matrix(X)

        scores = []

        for c in self.classes_:

            idx = self.class_indices_[c]

            class_score = np.mean(
                kernel[:, idx],
                axis=1
            )

            scores.append(class_score)

        scores = np.column_stack(scores)

        # Normalize
        denominator = np.sum(
            scores,
            axis=1,
            keepdims=True
        )

        denominator = np.maximum(
            denominator,
            1e-12
        )

        return scores / denominator

