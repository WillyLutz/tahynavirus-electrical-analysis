import datetime
import shutil
import time

import pandas as pd
import os
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import signal_processing as spr
import data_processing as dpr
import machine_learning as ml
import seaborn as sns
import PATHS as P
import data_processing as dp
from firelib.firelib import firefiles as ff, firelearn as fl
import pickle
import forestci as fci
from pathlib import Path

pd.set_option('display.max_columns', None)
import complete_procedures as cp
import get_plots as gp


def main():
    ni_nirg27_targets = ("NI", "NI+RG27",)
    training_targets_ni_and_rg27_condition = ("NI", "TAHV",)
    training_targets_without_rg27_impact = ("NI", "TAHV", )
    testing_targets_without_rg27_impact = ("NI", "TAHV", "NI+RG27")

    record = "T=48H"

    # getting features importances between NI and NI+RG27
    df_ni_ni27 = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK, ], targets=ni_nirg27_targets,
                                                  to_include=("freq_50hz_sample", record,), to_exclude=("TTX",),
                                                  verbose=False, save=False, )
    clf_ni_ni27 = ml.train_model_from_dataset(df_ni_ni27, scores=False)
    idx_ni_ni27, importance_ni_ni27 = ml.get_features_of_interest_from_trained_model(clf_ni_ni27, percentage=1,
                                                                                     title=f"features importance NI_NI+RG27 {record}")

    # getting the features importance between NI and the target we want to reduce RG27 impact
    df_with_rg27_impact = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK, ], targets=training_targets_ni_and_rg27_condition,
                                                        to_include=("freq_50hz_sample", record,), to_exclude=("TTX",),
                                                        verbose=False, save=False, )
    clf_with_rg27_impact = ml.train_model_from_dataset(df_with_rg27_impact, scores=False)
    idx_with_rg27_impact, importance_with_rg27_impact = ml.get_features_of_interest_from_trained_model(clf_with_rg27_impact, percentage=1,
                                                                                       title=f"features importance {training_targets_ni_and_rg27_condition} {record}")

    # subtract ni_ni27 importance from the importances we want to reduce rg27 impact
    subtract_importances = []
    for i in range(len(importance_ni_ni27)):
        res = importance_with_rg27_impact[i] - importance_ni_ni27[i]
        if res < 0:
            res = 0
        subtract_importances.append(res)

    features_idx_without_rg27_impact = []
    for i in range(len(subtract_importances)):
        if subtract_importances[i] > 0:
            features_idx_without_rg27_impact.append(i)

    # train model with the most important features (supposedly without rg27 interference)

    whole_training_df_without_rg27_impact = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK],
                                                                targets=training_targets_without_rg27_impact,
                                                                to_include=("freq_50hz_sample", "T=48H",),
                                                                to_exclude=("TTX",),
                                                                verbose=False,
                                                                save=False, )

    highest_training_df_without_rg27_impact = dpr.make_highest_features_dataset_from_complete_dataset(features_idx_without_rg27_impact, whole_training_df_without_rg27_impact)
    clf_without_rg27_impact = ml.train_model_from_dataset(highest_training_df_without_rg27_impact, scores=False)
    freed_importance = clf_without_rg27_impact.feature_importances_

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='all', sharex='all')
    axes[0, 0].bar(x=[x for x in range(len(importance_ni_ni27))], height=importance_ni_ni27)
    axes[0, 0].set_title(f"classes: {ni_nirg27_targets}")
    axes[0, 1].bar(x=[x for x in range(len(importance_with_rg27_impact))], height=importance_with_rg27_impact)
    axes[0, 1].set_title(f"classes: {training_targets_ni_and_rg27_condition}")
    axes[1, 0].bar(x=[x for x in range(len(subtract_importances))], height=subtract_importances)
    axes[1, 0].set_title(f"subtract of {ni_nirg27_targets} to {training_targets_ni_and_rg27_condition}")
    arranged_freed_importances = [0 for x in range(len(subtract_importances))]
    feature_idx = 0
    for i in range(len(arranged_freed_importances)):
        if i in features_idx_without_rg27_impact:
            arranged_freed_importances[i] = freed_importance[feature_idx]
            feature_idx += 1
    axes[1, 1].bar(x=[x for x in range(len(arranged_freed_importances))], height=arranged_freed_importances)
    axes[1, 1].set_title("retrained model on subtraction result")

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))
    xticks = [x for x in range(0, 300, 50)]
    new_ticks = [hertz[x] for x in xticks]
    xticks.append(300)
    new_ticks.append(5000)
    for axs in axes:
        for ax in axs:
            ax.set_xticks(xticks, new_ticks)
            ax.set_ylabel("Relative importance [AU]")
            ax.set_xlabel("Frequency-like features [Hz]")
    plt.suptitle(f"feature importance and RG27 impact subtraction to {training_targets_ni_and_rg27_condition} at {record} ")
    plt.savefig(os.path.join(P.RESULTS,f"feature importance and RG27 impact subtraction to {training_targets_ni_and_rg27_condition} at {record}.png" ), dpi=600)

    whole_testing_df_without_rg27_impact = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK],
                                                                 targets=testing_targets_without_rg27_impact,
                                                                 to_include=("freq_50hz_sample", "T=48H",),
                                                                 to_exclude=("TTX",),
                                                                 verbose=False,
                                                                 save=False, )

    highest_testing_df_without_rg27_impact = dpr.make_highest_features_dataset_from_complete_dataset(features_idx_without_rg27_impact, whole_testing_df_without_rg27_impact)
    ml.test_model(clf_without_rg27_impact, highest_testing_df_without_rg27_impact, training_targets=training_targets_without_rg27_impact, testing_targets=testing_targets_without_rg27_impact,
                  show=True, save=True, commentary="RG27 effect freed")




def test_by_target():
    recording_times = ["T=0MIN", "T=30MIN", "T=48H", "T=96H"]
    training_targets = ("NI", "TAHV", "TAHV+RG27",)
    testing_targets = ("NI", "TAHV", "TAHV+RG27", "NI+RG27",)

    df = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK, ],
                                          targets=training_targets,
                                          to_include=("freq_50hz_sample", "T=48H",),
                                          to_exclude=("TTX",),
                                          verbose=False,
                                          save=False, )

    clf = ml.train_model_from_dataset(df, scores=False)
    if training_targets != testing_targets:
        df = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK],
                                              targets=testing_targets,
                                              to_include=("freq_50hz_sample", "T=48H",),
                                              to_exclude=("TTX",),
                                              verbose=False,
                                              save=False, )
    ml.test_model(clf, df, verbose=False, save=True, show=True, training_targets=training_targets,
                  testing_targets=testing_targets, commentary="")


def test_from_scratch():
    df24 = dpr.make_dataset_from_freq_files(parent_directories=P.NORG27,
                                            to_include=("freq_50hz_sample", "T=0MIN"),
                                            to_exclude=("TTX",),
                                            verbose=True,
                                            save=False, )
    print(df24)
    clf = ml.train_model_from_dataset(df24)
    foi = ml.get_features_of_interest_from_trained_model(clf, percentage=0.05)
    del clf
    print(foi)
    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(foi, df24)
    clf = ml.train_model_from_dataset(hdf24)

    scores = ml.test_model(clf, hdf24, verbose=True)
    print(scores)


print(datetime.datetime.now())
main()

"""
    function(param1="")

        text

        Parameters
        ----------
        param1 : type, optional, default: 'a'
            text text text text text text text text text text text
            text text text

            .. versionadded:: 1.0.0

        Returns
        -------
        out : type
            text

    """