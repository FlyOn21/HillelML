from typing import List, Tuple

import numpy as np
import pandas as pd


def euclidean_distance(point1: np.array, point2: np.array) -> float:
    """Compute euclidean distance between two points.
    
    Args:
        point1: first point
        point2: second point
    
    Returns:
        euclidean distance between two points
    """
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


class KNN:
    """K Nearest Neighbors classifier."""
    def __init__(self, k: int) -> None:
        self._X_train = None
        self._y_train = None
        self.k = k  # number of neighbors to consider

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Predict target values for test data.

        Args:
            X_test: test data
            verbose: print progress during prediction,
                default is `False`.
        
        Returns:
            predicted target values
        """
        n = X_test.shape[0]
        y_pred = np.empty(n, dtype=self._y_train.dtype)

        for i in range(n):

            distances = np.zeros(len(self._X_train))
            for j in range(len(self._X_train)):
                distances[j] = euclidean_distance(self._X_train[j], X_test[i])
            
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self._y_train[k_indices]

            y_pred[i] = np.bincount(k_nearest_labels).argmax()

            if verbose:
                print(f"Predicted {i+1}/{n} samples", end="\r")

        if verbose:
            print("")
        return y_pred


def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split dataset into k folds.

    Args:
        X: dataset features
        y: dataset target
        k: number of folds
    
    Returns:
        list of tuples (X_train, y_train, X_test, y_test)
    """
    n_samples = X.shape[0]
    fold_size = n_samples // k
    folds = []  # container with results

    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size

        X_test = X[start_idx:end_idx]
        y_test = y[start_idx:end_idx]

        X_train = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
        y_train = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)

        folds.append((X_train, y_train, X_test, y_test))

    return folds


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score.
    
    Args:
        y_true: true target values
        y_pred: predicted target values
    
    Returns:
        accuracy score
    """

    n_samples = y_true.shape[0]
    correct_predictions = 0

    for true_val, pred_val in zip(y_true, y_pred):
        if true_val == pred_val:
            correct_predictions += 1

    accuracy = correct_predictions / n_samples
    return accuracy



def main() -> None:
    training_data = pd.read_csv("data/train.csv")[:1000]
    testing_data = pd.read_csv("data/test.csv")

    X = training_data.iloc[:, 1:].values
    y = training_data.iloc[:, 0].values
    print("Training data:", X.shape, y.shape)

    X_test = testing_data.iloc[:, 1:].values
    y_test = testing_data.iloc[:, 0].values
    print("    Test data:", X_test.shape, y_test.shape)

    k = 10  # NOTE: not the best choice for k
    print(f" KNN with k = {k}")

    num_folds = 4
    for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val, verbose=True)
        accuracy = evaluate_accuracy(y_val, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

    # TODO: compute accuracy on test data and compare results with cross validation scores



if __name__ == "__main__":
    main()
