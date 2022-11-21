import os
import pickle
import time

import forestci
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_processing as dpr
import data_analysis as dan
from firelib.firelib import firefiles as ff
import PATHS as P
import sys


def train_model_from_dataset(dataset, model_save_path="", save=False, scores=True):
    """
    Train an RFC model from a dataset and save the model.

    :param dataset: The dataset to train the model.
    :param model_save_path: Path to save the model
    :return: the trained RFC model
    """
    # todo : documentation

    clf = RandomForestClassifier(n_estimators=1000)
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf.fit(X_train, y_train)
    if scores:
        print(clf.score(X_test, y_test))
    clf.feature_names = [x for x in X.columns]
    if save:
        pickle.dump(clf, open(model_save_path, "wb"))
    return clf


def get_feature_of_interest(timepoint, path, detection_factor=2.0, plot=True, by_percentage=False, percentage=0.05):
    dataset = pd.read_csv(path)
    # training
    print("learning")
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    model_directory = P.MODELS
    ff.verify_dir(model_directory)
    importances_over_iterations = []
    std_over_iterations = []
    for i in range(10):
        clf = RandomForestClassifier(n_estimators=1000)
        clf.fit(X, y)
        mean = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        importances_over_iterations.append(mean)

    arrays = [np.array(x) for x in importances_over_iterations]
    mean_importances_over_iterations = [np.mean(k) for k in zip(*arrays)]
    std_arrays = [np.array(x) for x in importances_over_iterations]
    std_importances_over_iterations = [np.std(k) for k in zip(*std_arrays)]

    low_std = []
    for i in range(len(mean_importances_over_iterations)):
        low_std.append(mean_importances_over_iterations[i] - std_importances_over_iterations[i])
    high_std = []
    for i in range(len(mean_importances_over_iterations)):
        high_std.append(mean_importances_over_iterations[i] + std_importances_over_iterations[i])

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))

    whole_mean = np.mean(mean_importances_over_iterations)
    whole_std = np.std(mean_importances_over_iterations)

    high_mean_thresh = whole_mean + whole_std * detection_factor
    low_mean_thresh = whole_mean - whole_mean
    factor_mean_thresh = 1
    if plot:
        fig, ax = plt.subplots()
        ax.plot(hertz, mean_importances_over_iterations, color="red", linewidth=0.5)
        ax.fill_between(hertz, low_std, high_std, facecolor="blue", alpha=0.5)

        ax.axhline(y=whole_mean, xmin=0, xmax=300, color="black", linewidth=0.5)
        ax.fill_between(hertz, low_mean_thresh * factor_mean_thresh, high_mean_thresh * factor_mean_thresh,
                        facecolor="black", alpha=0.3)

        idx = []
        for i in range(len(mean_importances_over_iterations) - 1):
            value1 = mean_importances_over_iterations[i]
            value2 = mean_importances_over_iterations[i + 1]

            if value1 >= high_mean_thresh * factor_mean_thresh >= value2 or value1 <= high_mean_thresh * factor_mean_thresh <= value2:
                idx.append(hertz[i])

        for x in idx:
            ax.axvline(x=x, color="green", linewidth=0.5)

        title = f"features of interest at {timepoint} with {detection_factor} factor detection"
        plt.show()

    if by_percentage:
        n = int(percentage * len(mean_importances_over_iterations))
        idx_foi = sorted(range(len(mean_importances_over_iterations)),
                         key=lambda i: mean_importances_over_iterations[i], reverse=True)[:n]
        return idx_foi
    else:
        idx_foi = []
        for i in range(len(mean_importances_over_iterations) - 1):
            if mean_importances_over_iterations[i] >= high_mean_thresh * factor_mean_thresh:
                idx_foi.append(i)

        return idx_foi


def get_features_of_interest_from_trained_model(clf, percentage=0.05, show=False, save=False, title=""):
    """
    Only for model not trained on restricted features.

    :param clf:
    :param percentage:
    :return:
    """
    importances_over_iterations = []
    std_over_iterations = []
    for i in range(10):
        mean = np.mean([tree.feature_importances_ for tree in clf.estimators_], axis=0)

        importances_over_iterations.append(mean)

    arrays = [np.array(x) for x in importances_over_iterations]
    mean_importances_over_iterations = [np.mean(k) for k in zip(*arrays)]
    std_arrays = [np.array(x) for x in importances_over_iterations]
    std_importances_over_iterations = [np.std(k) for k in zip(*std_arrays)]

    low_std = []
    for i in range(len(mean_importances_over_iterations)):
        low_std.append(mean_importances_over_iterations[i] - std_importances_over_iterations[i])
    high_std = []
    for i in range(len(mean_importances_over_iterations)):
        high_std.append(mean_importances_over_iterations[i] + std_importances_over_iterations[i])

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))
    n = int(percentage * len(mean_importances_over_iterations))
    idx_foi = sorted(range(len(mean_importances_over_iterations)),
                     key=lambda i: mean_importances_over_iterations[i], reverse=True)[:n]

    plt.bar([x for x in range(300)], mean_importances_over_iterations, color="blue", )

    xticks = [x for x in range(0, 300, 50)]
    new_ticks = [hertz[x] for x in xticks]
    xticks.append(300)
    new_ticks.append(5000)
    plt.xticks(xticks, new_ticks)
    plt.title(title)
    if save:
        plt.savefig(os.path.join(P.RESULTS, title + ".png"))
    # plt.fill_between(hertz, low_std, high_std, facecolor="blue", alpha=0.5)
    if show:
        plt.show()
    return idx_foi


def test_model(clf, dataset, verbose=False, show=True, training_targets=(),
               testing_targets=(),
               save=False, commentary=""):
    """
    Test a model on a dataset.

    :param clf: The model
    :param dataset: dataset used for the testing
    :param iterations: number of iteration for testing
    :return: scores
    """
    # todo: make testing over iterations
    if not testing_targets:
        testing_targets = training_targets
        
    CORRESPONDANCE = {}
    target_id = 0
    for t in training_targets:
        if t not in CORRESPONDANCE:
            CORRESPONDANCE[t] = target_id
            target_id += 1
    for t in testing_targets:
        if t not in CORRESPONDANCE:
            CORRESPONDANCE[t] = target_id
            target_id += 1
            

    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    if verbose:
        progress = 0
        sys.stdout.write(f"\rTesting model: {progress}%")
        sys.stdout.flush()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # get predictions and probabilities

    title = f"train {training_targets}-test {testing_targets}"
    matrix = np.zeros((len(training_targets), len(testing_targets)))
    probabilities_matrix = np.empty((len(training_targets), len(testing_targets)), dtype=object)
    mixed_labels_matrix = np.empty((len(training_targets), len(testing_targets))).tolist()
    mean_probabilities_matrix = np.empty((len(training_targets), len(testing_targets)))

    # Initializing the matrix containing the probabilities
    for i in range(len(probabilities_matrix)):
        for j in range(len(probabilities_matrix[i])):
            probabilities_matrix[i][j] = []

    # Making predictions and storing the results in predictions[]
    predictions = []
    for i in X_test.index:
        row = X_test.loc[i, :]
        y_pred = clf.predict([row])[0]
        proba_class = clf.predict_proba([row])[0]
        predictions.append((y_pred, proba_class))

    #
    targets = []
    for i in y_test.index:
        targets.append(y_test.loc[i])

    # Building the confusion matrix
    for i in range(len(targets)):
        y_true = targets[i]
        y_pred = predictions[i][0]
        y_proba = max(predictions[i][1])
        matrix[CORRESPONDANCE[y_pred]][CORRESPONDANCE[y_true]] += 1 # todo: attention, si on a que 2 targets dont +RG27, on prend des valeurs
        # todo: y_pred/true à 2 ou 3, ce qui est superieur aux dimensions de la matrice
        # todo:

        probabilities_matrix[CORRESPONDANCE[y_pred]][CORRESPONDANCE[y_true]].append(y_proba)
    # averaging the probabilities
    for i in range(len(probabilities_matrix)):
        for j in range(len(probabilities_matrix[i])):
            mean_probabilities_matrix[i][j] = np.mean(probabilities_matrix[i][j])

    # mixing count and probabilities fpr displaying
    for i in range(len(probabilities_matrix)):
        for j in range(len(probabilities_matrix[i])):
            np.nan_to_num(matrix[i][j])
            np.nan_to_num(mean_probabilities_matrix[i][j])
            mixed_labels_matrix[i][j] = str(int(matrix[i][j])) + "\nCUP=" + str(
                round(mean_probabilities_matrix[i][j], 3))

    # plotting
    fig, ax = plt.subplots(1, 1, figsize=(7/4*len(testing_targets), 6/4*len(training_targets)))

    fig.suptitle("")
    sns.heatmap(ax=ax, data=matrix, annot=mixed_labels_matrix, fmt='')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_ylabel("The input is classified as")
    ax.set_xlabel("The input is")
    ax.set_xticks([CORRESPONDANCE[x] + 0.5 for x in testing_targets], testing_targets)
    ax.set_yticks([CORRESPONDANCE[x] + 0.5 for x in training_targets], training_targets)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(P.RESULTS, "Confusion matrix-" + title + ".png"))
    if show:
        plt.show()

    # all_metrics.append(((positive_class, negative_class), (tp, tn, fp, fn)))
