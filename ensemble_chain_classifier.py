import numpy as np
import random

## Ready-made Machine Learning algos to be used in the custom classifiers
from sklearn.tree import DecisionTreeClassifier

## Tools for developing the custom Classifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import clone


from chain_classifier import ChainClassifier


class EnsembleChainClassifier(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    n : int, the number of ensemble members to create. each member is a chain classifier.
    base_classifier : sklearn Estimator, default=sklearn.tree.DecisionTreeClassifier()
    threshold : float, value above which aggregated predictions are considered significant.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    classifiers_ : dict
        Mapping of each class label to the chain classifier that has been trained to predict that particular label.
    """
    def __init__(self, n=10, base_classifier=DecisionTreeClassifier(), threshold=0.5):
        self.base_classifier = base_classifier
        self.n = n
        self.threshold = threshold
    
    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, multi_output=True)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        n_rows, n_features = X.shape
        _, n_targets = y.shape

        self.classifiers_ = {}
        columns = [i for i in range(n_targets)]
        for i in range(self.n):
            cc = ChainClassifier(self.base_classifier, ordering=list(columns))
            sample = np.random.choice(n_rows, size=n_rows//3)
            self.classifiers_[i] = cc.fit(X[sample], y[sample])
            # Shuffle the column ordering for the next chain classifier
            random.shuffle(columns)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_', 'classifiers_'])

        # Input validation
        X = check_array(X)
        n_samples, _ = X.shape
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        for j in range(self.n):
            pred = self.classifiers_[j].predict(X)
            predictions = predictions + pred

        predictions = predictions / self.n
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < self.threshold] = 0
        return predictions

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        probabilities : 2d-array
            Probability of a query belonging to each class.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        _, n_features = self.X_.shape
        probabilities = np.zeros((X.shape[0], len(self.classes_)))
        for i, chain in self.classifiers_.items():
            proba = chain.predict_proba(X)
            probabilities += proba
        return probabilities / probabilities.sum(axis=1, keepdims=True)
