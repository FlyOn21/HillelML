import os
import re
import tempfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import timeit
from sklearn.model_selection import KFold


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

            k_indices = np.argsort(distances)[: self.k]
            k_nearest_labels = self._y_train[k_indices]

            y_pred[i] = np.bincount(k_nearest_labels).argmax()

            if verbose:
                print(f"Predicted {i + 1}/{n} samples", end="\r")

        if verbose:
            print(f"")
        return y_pred


# def kfold_cross_validation(X: np.ndarray, y: np.ndarray, k: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
#     """Split dataset into k folds.
#
#     Args:
#         X: dataset features
#         y: dataset target
#         k: number of folds
#
#     Returns:
#         list of tuples (X_train, y_train, X_test, y_test)
#     """
#     n_samples = X.shape[0]
#     fold_size = n_samples // k
#     folds = []  # container with results
#
#     for i in range(k):
#         start_idx = i * fold_size
#         end_idx = (i + 1) * fold_size
#
#         X_test = X[start_idx:end_idx]
#         y_test = y[start_idx:end_idx]
#
#         X_train = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
#         y_train = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)
#
#         folds.append((X_train, y_train, X_test, y_test))
#     return folds


def kfold_cross_validation(
    X: np.ndarray, y: np.ndarray, k: int
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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
    kf = KFold(n_splits=k)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append((X_train, y_train, X_test, y_test))
    # for i in folds:
    #     print(i[0].shape, i[1].shape, i[2].shape, i[3].shape)
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


def training_model_KNN(X: np.ndarray, y: np.ndarray, num_folds: int, k: int) -> List[float]:
    """
    Train a k-Nearest Neighbors (KNN) model using k-fold cross validation.

    Parameters:
    X (np.ndarray): The training data.
    y (np.ndarray): The training labels.
    num_folds (int): The number of folds for cross-validation.
    k (int): The number of nearest neighbors to consider in the KNN algorithm.

    Returns:
    List[float]: A list of accuracy scores for each fold of the cross-validation.
    """
    accuracy_list = []
    for X_train, y_train, X_val, y_val in kfold_cross_validation(X, y, k=num_folds):
        model = KNN(k=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val, verbose=True)
        accuracy = evaluate_accuracy(y_val, y_pred)
        print(f"Accuracy training: {accuracy:.2f}")
        accuracy_list.append(accuracy)
    return accuracy_list


def testing_model_KNN(X_test: np.ndarray, y_test: np.ndarray, k: int) -> float:
    """
    Test a k-Nearest Neighbors (KNN) model.

    Parameters:
    X_test (np.ndarray): The test data.
    y_test (np.ndarray): The test labels.
    k (int): The number of nearest neighbors to consider in the KNN algorithm.

    Returns:
    float: The accuracy score of the model on the test data.
    """
    model = KNN(k=k)
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test, verbose=True)
    accuracy = evaluate_accuracy(y_test, y_pred)
    print(f"Accuracy test: {accuracy:.2f}")
    return accuracy


def training_and_testing_model_KNN(
    X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, k: int, num_folds: int
) -> Tuple[float, float, float]:
    """
    Train and test a k-Nearest Neighbors (KNN) model, then calculate the difference in accuracy.

    This function trains a KNN model on given data using k-fold cross-validation, tests the model on separate test
    data, and calculates the difference between the average training accuracy and the test accuracy.

    Parameters:
    X (np.ndarray): The training data.
    y (np.ndarray): The training labels.
    X_test (np.ndarray): The test data.
    y_test (np.ndarray): The test labels.
    k (int): The number of nearest neighbors to consider in the KNN algorithm.
    num_folds (int): The number of folds for cross-validation.

    Returns:
    Tuple[float, float, float]: A tuple containing the average training accuracy, the test accuracy, and the difference
    between these two accuracies.
    """
    training_accuracy_result: List[float] = training_model_KNN(X, y, num_folds, k)
    average_training_accuracy = np.mean(training_accuracy_result)
    test_accuracy_result: float = testing_model_KNN(X_test, y_test, k)
    diff_accuracy = np.abs(average_training_accuracy - test_accuracy_result)
    return average_training_accuracy, test_accuracy_result, diff_accuracy


def main(num_folds: int = 4, write_result: bool = False) -> None:
    dir_path = os.path.dirname(os.path.abspath(__file__))
    training_data = pd.read_csv(os.path.join(dir_path, "data/train.csv"))[:1000]
    testing_data = pd.read_csv(os.path.join(dir_path, "data/test.csv"))

    X = training_data.iloc[:, 1:].values
    y = training_data.iloc[:, 0].values
    print("Training data:", X.shape, y.shape)

    X_test = testing_data.iloc[:, 1:].values
    y_test = testing_data.iloc[:, 0].values
    print("    Test data:", X_test.shape, y_test.shape)

    best_k = 0
    diff = None

    try:
        with open(os.path.join(dir_path, "README.md"), "r", encoding="utf-8") as f:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf8") as tmp_file:
                file_path = tmp_file.name
                lines = f.readlines()
                for line in lines:
                    if re.search(r"\d \|", line):
                        k = int(line.split()[0])
                        print(f" KNN with k = {k}")
                        training_accuracy_result, test_accuracy_result, diff_accuracy = training_and_testing_model_KNN(
                            X, y, X_test, y_test, k, num_folds
                        )
                        line = (
                            f"{k} | Test: {test_accuracy_result:.2f} # "
                            f"Training: {np.mean(training_accuracy_result):.2f} # "
                            f"Difference: {diff_accuracy:.2f}\n"
                        )
                        if not best_k:
                            best_k = k
                            print(f"11111New best k = {k}")
                            diff = diff_accuracy
                        else:
                            if diff > diff_accuracy:
                                print(f"New best k = {k}")
                                best_k = k
                                diff = diff_accuracy
                        tmp_file.write(line)
                    else:
                        if line.startswith("## Test with different number of folds"):
                            break
                        tmp_file.write(line)
                test_switch_num_folds = [i for i in range(2, 13)]
                tmp_file.write("\n")
                tmp_file.write(f"## Test with different number of folds with best k={best_k}\n")
                tmp_file.write("\n")
                tmp_file.write("Number of folds | Accuracy \n")
                tmp_file.write("--------------- | --------\n")
                for num_fold in test_switch_num_folds:
                    training_accuracy_result, test_accuracy_result, diff_accuracy = training_and_testing_model_KNN(
                        X, y, X_test, y_test, best_k, num_fold
                    )
                    line = (
                        f"{num_fold} | Test: {test_accuracy_result:.2f} # "
                        f"Training: {np.mean(training_accuracy_result):.2f} # "
                        f"Difference: {diff_accuracy:.2f}\n"
                    )
                    tmp_file.write(line)
        if write_result:
            with open(os.path.join(dir_path, "README.md"), "w", encoding="utf-8") as f:
                with open(file_path, "r", encoding="utf-8") as tmp_file:
                    lines = tmp_file.readlines()
                    for line in lines:
                        f.write(line)
    finally:
        os.remove(file_path)


if __name__ == "__main__":
    main(write_result=True, num_folds=4)
