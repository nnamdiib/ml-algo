import numpy as np

## Ready-made Machine Learning algos to be used in the custom classifiers
from sklearn.tree import DecisionTreeClassifier

## Tools for developing the custom Classifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import clone
from imblearn import under_sampling  # For undersampling


class BinaryRelevanceClassifier(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    base_classifier : sklearn Estimator, default=sklearn.tree.DecisionTreeClassifier()
    balance_classes : Boolean, default=False. If set to True, each label will be undersampled to balance the
                      the values in each column (0 or 1)
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
    def __init__(self, base_classifier=DecisionTreeClassifier(), balance_classes=False):
        self.base_classifier = base_classifier
        self.balance_classes = balance_classes

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

        # Set up the Undersampler
        under_sampler = under_sampling.RandomUnderSampler(random_state=2)
        # Dict to store our trained classifiers
        self.classifiers_ = {}
        for i in range(n_targets):
            classifier = clone(self.base_classifier)
            X_to_fit = X
            y_to_fit = y[:, i]  # Select the i-th label
            if self.balance_classes:
                # Under sample the training features and labels
                X_to_fit, y_to_fit = under_sampler.fit_sample(X, y[:, i])
            self.classifiers_[i] = classifier.fit(X_to_fit, y_to_fit)

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
        predictions = []
        for j in range(len(self.classes_)):
            predictions.append(self.classifiers_[j].predict(X))
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
        n_classes = len(self.classes_)
        probabilities = []
        for j in range(n_classes):
            # Append only the 'Yes' probability column to the final matrix
            probabilities.append(self.classifiers_[j].predict_proba(X)[:, 1])
        return np.column_stack(probabilities)
