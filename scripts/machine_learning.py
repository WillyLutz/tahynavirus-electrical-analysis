import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import PATHS as P


def train_RFC_from_dataset(dataset, save_filename=""):
    """
    train_model_from_dataset(dataset, model_save_path="", save=False, scores=True):

        Train a Random Forest Classifier model from an already formatted dataset.

        Parameters
        ----------
        dataset : pandas Dataframe
            a pandas Dataframe where each row is an entry for a machine
            learning model. Has a last column as 'target' containing
            the target value for each entry.

        save_filename: str, optional, default:''
            name of the saved file. If empty, does not save the file.

        scores: bool, optional, default: False
            Whether to return the clf.score(X_test, y_test) sklearn
            metric or not.

            .. versionadded:: 1.0.0

        Returns
        -------
        out : RandomForestClassifier
            a trained random forest classifier.
    """

    clf = RandomForestClassifier(n_estimators=1000)
    X = dataset[dataset.columns[:-1]]
    y = dataset["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf.fit(X_train, y_train)
    if save_filename:
        pickle.dump(clf, open(os.path.join(P.MODELS, save_filename), "wb"))
    return clf, clf.score(X_test, y_test)


def get_top_features_from_trained_model(clf, percentage=0.05, show=False, save=False, title=""):
    """
        get_features_of_interest_from_trained_model(clf, percentage=0.05, show=False, save=False, title=""):

            select to top n% feature sorted by highest importance, of a trained Random Forest Classtifier model.

            Parameters
            ----------
            clf : RandomForestClassifier
                a trained model

            percentage: float, optional, default: 0.05
                proportion of the most important features to keep

            show: bool, optional, default: False
                Whether to show a plot of the model feature importance or not.

            save: bool, optional, default: False
                Whether to save a plot of the model feature importance or not.

            title: str, optional, default: ''
                the title to give the plot and name of the resulting file if save if True.

                .. versionadded:: 1.0.0

            Returns
            -------
            out : tuple of lists
                first element: list of the indexes of the most important features
                second element: importance (values) corresponding to the indexes.
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
    plt.ylabel("Relative importance [AU]")
    plt.xlabel("Frequency-like features [Hz]")
    plt.title(title)
    if save:
        plt.savefig(os.path.join(P.RESULTS, title + ".png"))
    # plt.fill_between(hertz, low_std, high_std, facecolor="blue", alpha=0.5)
    if show:
        plt.show()
    plt.close()
    return idx_foi, mean_importances_over_iterations


def test_model(clf, dataset, training_targets, verbose=False, show=True, testing_targets=(), save=False, commentary=""):
    """
        test_model(clf, dataset, training_targets, verbose=False, show=True, testing_targets=(), save=False, commentary=""):

            Test an already trained Random forest classifier model,
            resulting in a confusion matrix. The test can be done
            on targets_labels different from the targets_labels used for training
            the model.

            Parameters
            ----------
            clf: RandomForestClassifier
                the trained model.
            dataset:  pandas Dataframe.
                Dataframe containing the data used for testing the
                model. The rows are the entries, and the columns are
                the features on which the model has been trained.
                The last column is 'status' containing the labels
                of the targets_labels for each entry.
            training_targets: tuple of str

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

    print(CORRESPONDANCE)
    X = dataset[dataset.columns[:-1]]
    y = dataset["status"]

    if verbose:
        progress = 0
        sys.stdout.write(f"\rTesting model: {progress}%")
        sys.stdout.flush()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # get predictions and probabilities

    title = f"train {training_targets}-test {testing_targets} {commentary}"
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
        matrix[CORRESPONDANCE[y_pred]][CORRESPONDANCE[y_true]] += 1

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
    fig, ax = plt.subplots(1, 1, figsize=(7 / 4 * len(testing_targets), 6 / 4 * len(training_targets)))

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


def fit_pca(dataframe, n_components=3):
    """
    fit_pcs(dataframe, n_components):

        fit a Principal Component Analysis and return its instance and dataset.

        Parameters
        ----------
        dataframe: DataFrame
            The data on which the pca instance has to be fitted.
        n_components: int, optional, default: 3
            The number of components for the PCA instance.

        Returns
        -------
        out: tuple
            The first element is the PCA instance. The second
            element is the resulting dataframe.
    """
    features = dataframe.columns[:-1]
    x = dataframe.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features
    pca = PCA(n_components=n_components)
    principalComponent = pca.fit_transform(x)
    principal_component_columns = [f"principal component {i + 1}" for i in range(n_components)]

    principal_tahyna_Df = pd.DataFrame(data=principalComponent
                                       , columns=principal_component_columns)

    principal_tahyna_Df["label"] = dataframe["label"]

    return pca, principal_tahyna_Df, pca.explained_variance_ratio_


def apply_pca(pca: PCA, dataframe):
    """
    apply_pca(pca, dataframe):

        Transform data using an already fit PCA instance.

        Parameters
        ----------
        pca: PCA instance
            The fitted PCA instance from what the data will
            be transformed.
        dataframe: DataFrame
            The data to transform using an already fitted PCA.
            Must have a 'label' column.

        Returns
        -------
        out: DataFrame
            The transformed data.
    """
    features = dataframe.columns[:-1]
    x = dataframe.loc[:, features].values
    x = StandardScaler().fit_transform(x)  # normalizing the features
    transformed_ds = pca.transform(x)
    transformed_df = pd.DataFrame(data=transformed_ds, columns=[f"principal component {i+1}" for i in range(transformed_ds.shape[1])])
    transformed_df['label'] = dataframe['label']
    return transformed_df


def plot_pca(dataframe: pd.DataFrame, n_components=3, show=True, save=False, commentary="T=48H"):
    """
    plot_pca(dataframe, n_components, show=True, save=True):

        plot the result of PCA.

        Parameters
        ----------
        dataframe: DataFrame
            The data to plot. Must contain a 'label' column.
        n_components: int, optional, default: 3
            Number of principal components. Also, teh dimension
            of the graph. Must be equal to 2 or 3.
        show: bool, optional, default: True
            Whether to show the plot or not.
        save: bool, optional, default: False
            Whether to save the plot or not.
        commentary: str, optional, default: "T=48H"
            Any specification to include in the file name while saving.
    """
    targets = (list(set(dataframe["label"])))
    if show or save:
        if n_components == 2:
            plt.figure(figsize=(10, 10))
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=14)
            plt.xlabel(f'PC-1', fontsize=20)
            plt.ylabel(f'PC-2', fontsize=20)
            plt.title(f"Principal Component Analysis for TAHV infection", fontsize=20)
            colors = ['r', 'g', 'b', 'k']
            for target, color in zip(targets, colors):
                indicesToKeep = dataframe['label'] == target
                plt.scatter(dataframe.loc[indicesToKeep, 'principal component 1']
                            , dataframe.loc[indicesToKeep, 'principal component 2'], c=color, s=10)
            plt.legend(targets, prop={'size': 15})
        elif n_components == 3:
            plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.set_xlabel(f'PC-1', fontsize=20)
            ax.set_ylabel(f'PC-2', fontsize=20)
            ax.set_zlabel(f'PC-3', fontsize=20)
            colors = ['r', 'g', 'b', 'k']
            plt.title(f"Principal Component Analysis for TAHV infection", fontsize=20)
            for target, color in zip(targets, colors):
                indicesToKeep = dataframe['label'] == target
                x = dataframe.loc[indicesToKeep, 'principal component 1']
                y = dataframe.loc[indicesToKeep, 'principal component 2']
                z = dataframe.loc[indicesToKeep, 'principal component 3']
                ax.scatter3D(x, y, z, c=color, s=10)
            plt.legend(targets, prop={'size': 15})
    if save:
        if commentary:
            plt.savefig(os.path.join(P.RESULTS, f"PCA n={n_components} t={targets} {commentary}.png"))
        else:
            plt.savefig(os.path.join(P.RESULTS, f"PCA n={n_components} t={targets}.png"))

    if show:
        plt.show()
    plt.close()

