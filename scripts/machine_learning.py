import os
import pickle
import sys
from random import randint

import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from scipy.spatial import ConvexHull
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
import fiiireflyyy.firelearn as fl

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

            .. versionadded:: 1.0.0

        Returns
        -------
        out : tuple of size (1, 2).
            The first element is a trained random forest classifier.
            The second is its scores.
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
        out: tuple of shape (1, 3)
            The first element is the PCA instance. The second
            element is the resulting dataframe. The third is the
            explained variance ratios.
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
    transformed_df = pd.DataFrame(data=transformed_ds,
                                  columns=[f"principal component {i + 1}" for i in range(transformed_ds.shape[1])])
    transformed_df['label'] = dataframe['label']
    return transformed_df


def confidence_ellipse(x, y, ax, n_std=3.0, color='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      color=color, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_pca(dataframe: pd.DataFrame, **kwargs): # todo: to fiiireflyyy
    """
    plot the result of PCA.

    Parameters
    ----------
    dataframe: DataFrame
        The data to plot. Must contain a 'label' column.
    n_components: int, optional, default: 2
        Number of principal components. Also, teh dimension
        of the graph. Must be equal to 2 or 3.
    show: bool, optional, default: True
        Whether to show the plot or not.
    save: bool, optional, default: False
        Whether to save the plot or not.
    commentary: str, optional, default: "T=48H"
        Any specification to include in the file name while saving.
    points: bool, optional, default: True
        whether to plot the points or not.
    metrics: bool, optional, default: False
        Whether to plot the metrics or not
    savedir: str, optional, default: ""
        Directory where to save the resulting plot, if not empty.
    title: str, optional, defualt: ""
        The filename of the resulting plot. If empty,
        an automatic name will be generated.
    ratios: tuple of float, optional, default: ()
        the PCA explained variance ratio
    """

    options = {
        'n_components': 2,
        'show': True,
        'commentary': "",
        'points': True,
        'metrics': False,
        'savedir': "",
        'pc_ratios': [],
        'title': "",
        'ratios': ()

    }

    options.update(kwargs)
    targets = (list(set(dataframe["label"])))
    colors = ['g', 'b', 'r', 'k', 'sandybrown', 'deeppink', 'gray']
    if len(targets) > len(colors):
        n = len(targets) - len(colors) + 1
        for i in range(n):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

    label_params = {'fontsize': 20, "labelpad": 8}
    ticks_params = {'fontsize': 20, }
    if options['n_components'] == 2:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        plt.xticks(**ticks_params)
        plt.yticks(**ticks_params)
        xlabel = f'Principal Component-1 ({options["ratios"][0]}%)'
        ylabel = f'Principal Component-2 ({options["ratios"][1]}%)'
        if len(options['pc_ratios']):
            xlabel += f" ({round(options['pc_ratios'][0] * 100, 2)}%)"
            ylabel += f" ({round(options['pc_ratios'][1] * 100, 2)}%)"

        plt.xlabel(xlabel, **label_params)
        plt.ylabel(ylabel, **label_params)

        for target, color in zip(targets, colors):
            indicesToKeep = dataframe['label'] == target
            x = dataframe.loc[indicesToKeep, 'principal component 1']
            y = dataframe.loc[indicesToKeep, 'principal component 2']
            if options['points']:
                alpha = 1
                if options['metrics']:
                    alpha = .2
                plt.scatter(x, y, c=color, s=10, alpha=alpha, label=target)
            if options['metrics']:
                plt.scatter(np.mean(x), np.mean(y), marker="+", color=color, linewidth=2, s=160)
                fl.confidence_ellipse(x, y, ax, n_std=1.0, color=color, fill=False, linewidth=2)

        def update(handle, orig):
            handle.update_from(orig)
            handle.set_alpha(1)

        plt.legend(prop={'size': 25}, handler_map={PathCollection: HandlerPathCollection(update_func=update),
                                                   plt.Line2D: HandlerLine2D(update_func=update)})
    elif options['n_components'] == 3:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

        xlabel = f'Principal Component-1 ({options["ratios"][0]}%)'
        ylabel = f'Principal Component-2 ({options["ratios"][1]}%)'
        zlabel = f'Principal Component-3 ({options["ratios"][2]}%)'
        if len(options['pc_ratios']):
            xlabel += f" ({round(options['pc_ratios'][0] * 100, 2)}%)"
            ylabel += f" ({round(options['pc_ratios'][1] * 100, 2)}%)"
            zlabel += f" ({round(options['pc_ratios'][2] * 100, 2)}%)"

        ax.set_xlabel(xlabel, **label_params)
        ax.set_ylabel(ylabel, **label_params)
        ax.set_zlabel(zlabel, **label_params)
        for target, color in zip(targets, colors):
            indicesToKeep = dataframe['label'] == target
            x = dataframe.loc[indicesToKeep, 'principal component 1']
            y = dataframe.loc[indicesToKeep, 'principal component 2']
            z = dataframe.loc[indicesToKeep, 'principal component 3']
            ax.scatter3D(x, y, z, c=color, s=10)
        plt.legend(targets, prop={'size': 18})

    if options['savedir']:
        if options["title"] == "":
            if options['commentary']:
                options["title"] += options["commentary"]

        plt.savefig(os.path.join(options['savedir'], options["title"] + ".png"), dpi=1200)

    if options['show']:
        plt.show()
    plt.close()

def fit_umap(dataframe, n_components=3):
    # Configure UMAP hyperparameters
    features = dataframe.columns[:-1]
    x = dataframe.loc[:, features].values
    reducer = UMAP(n_neighbors=100,
                   # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                   n_components=n_components,  # default 2, The dimension of the space to embed into.
                   metric='euclidean',
                   # default 'euclidean', The metric to use to compute distances in high dimensional space.
                   n_epochs=1000,
                   # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
                   learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
                   init='spectral',
                   # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                   min_dist=0.1,  # default 0.1, The effective minimum distance between embedded points.
                   spread=1.0,
                   # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                   low_memory=False,
                   # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                   set_op_mix_ratio=1.0,
                   # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                   local_connectivity=1,
                   # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                   repulsion_strength=1.0,
                   # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                   negative_sample_rate=5,
                   # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                   transform_queue_size=4.0,
                   # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                   a=None,
                   # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                   b=None,
                   # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                   random_state=42,
                   # default: None, If int, random_state is the seed used by the random number generator;
                   metric_kwds=None,
                   # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                   angular_rp_forest=False,
                   # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                   target_n_neighbors=-1,
                   # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                   # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
                   # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                   # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                   transform_seed=42,
                   # default 42, Random seed used for the stochastic aspects of the transform operation.
                   verbose=False,  # default False, Controls verbosity of logging.
                   unique=False,
                   # default False, Controls if the rows of your data should be uniqued before being embedded.
                   )

    # Fit and transform the data
    X_trans = reducer.fit_transform(x)
    X_dimension = [f"dimension {i + 1}" for i in range(n_components)]
    transformed_df = pd.DataFrame(data=X_trans, columns=X_dimension)

    transformed_df["label"] = dataframe["label"]

    return reducer, transformed_df


def apply_umap(umap, dataframe):
    """
       apply_umap(umap, dataframe):

           Transform data using an already fit UMAP instance.

           Parameters
           ----------
           umap: UMAP instance
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
    transformed_ds = umap.transform(x)
    transformed_df = pd.DataFrame(data=transformed_ds,
                                  columns=[f"dimension {i + 1}" for i in range(transformed_ds.shape[1])])
    transformed_df['label'] = dataframe['label']
    return transformed_df


def plot_umap(dataframe, n_component=3, save=False, show=True, commentary="T=48H"):
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(f'dimension-1', fontsize=20)
    ax.set_ylabel(f'dimension-2', fontsize=20)
    ax.set_zlabel(f'dimension-3', fontsize=20)
    targets = (sorted(list(set(dataframe["label"]))))
    colors = ['tomato', 'forestgreen', 'cornflowerblue', 'orange', 'sandybrown', 'maroon', 'black']
    for target, color in zip(targets, colors):
        indicesToKeep = dataframe['label'] == target
        x = dataframe.loc[indicesToKeep, dataframe.columns[0]]
        y = dataframe.loc[indicesToKeep, dataframe.columns[1]]
        z = dataframe.loc[indicesToKeep, dataframe.columns[2]]
        ax.scatter3D(x, y, z, c=color, s=10, label='bla')
    plt.legend(targets, prop={'size': 15})
    plt.title(f"Uniform Manifold Approximated Projection for TAHV infection", fontsize=20)

    if save:
        if commentary:
            plt.savefig(os.path.join(P.RESULTS, f"UMAP n={n_component} t={targets} {commentary}.png"))
        else:
            plt.savefig(os.path.join(P.RESULTS, f"UMAP n={n_component} t={targets}.png"))

    if show:
        plt.show()
    plt.close()
