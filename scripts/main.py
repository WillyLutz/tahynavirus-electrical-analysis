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
    # test_from_scratch()
    rg27 = [P.NORG27, P.RG27]
    recording_times = ["T=0MIN", "T=30MIN", "T=48H", "T=96H"]
    training_targets = ("NI", "TAHV",)
    testing_targets = ("NI", "TAHV",)

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


def test_with_loaded_model():
    clf = pickle.load(open(os.path.join(P.MODELS, "T=24H mixed organoids - base foi - no stachel.sav"), "rb"))
    df24 = dpr.make_dataset_from_freq_files(parent_directories=P.STACHEL,
                                            to_include=("freq_50hz_sample", "T=24H", "TTX",),
                                            to_exclude=("STACHEL",),
                                            verbose=False,
                                            save=False, )

    hdf24 = dpr.make_highest_features_dataset_from_complete_dataset(clf.feature_names, df24)

    scores = ml.test_model(clf, hdf24, verbose=False, )
    print(scores)


print(datetime.datetime.now())
main()
