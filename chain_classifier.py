import numpy as np

## Ready-made Machine Learning algos to be used in the custom classifiers
from sklearn.tree import DecisionTreeClassifier

## Tools for developing the custom Classifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import clone


class ChainClassifier(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    base_classifier : sklearn Estimator, default=sklearn.tree.DecisionTreeClassifier()
    ordering: list, default=None. ordering is the column order in which the base classifiers would be trained.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    classifiers_ : dict
        Mapping of each class label to the classifier that has been trained to predict that particular label.
    """
    def __init__(self, base_classifier=DecisionTreeClassifier(), ordering=None):
        self.base_classifier = base_classifier
        self.ordering = ordering

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
        if self.ordering and len(self.ordering) != n_targets:
            raise ValueError('Ordering size must match number of class labels')

        self.classifiers_ = {}
        X_augment = X
        ordering = self.ordering if self.ordering else [i for i in range(n_targets)]
        for i, col in enumerate(ordering):
            classifier = clone(self.base_classifier)
            if i > 0:
                # Add the previous target col as part of the current feature set.
                X_augment = np.c_[X_augment, y[:, ordering[i-1]]]
            y_fit = y[:, col]
            self.classifiers_[col] = classifier.fit(X_augment, y_fit)
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
        _, n_targets = self.y_.shape
        predictions = [-1] * n_targets
        X_augment = X
        ordering = self.ordering if self.ordering else (i for i in range(n_targets))

        for col in ordering:
            pred = self.classifiers_[col].predict(X_augment)
            predictions[col] = pred
            X_augment = np.c_[X_augment, pred]

        return np.column_stack(predictions)

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
        probabilities = [-1] * len(self.classes_)
        ordering = self.ordering if self.ordering else (i for i in range(len(self.classes_)))
        X_augment = X
        for col in ordering:
            pred = self.classifiers_[col].predict(X_augment)
            proba = self.classifiers_[col].predict_proba(X_augment)
            probabilities[col] = proba[:, 1]
            X_augment = np.c_[X_augment, pred]
        return np.column_stack(probabilities)
