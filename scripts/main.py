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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    train_targets = ("NI", "TAHV")
    test_targets = ("NI+RG27", "TAHV+RG27")
    record = "T=48H"
    train_dataset = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK, ], targets_labels=train_targets,
                                                     to_include=("freq_50hz_sample", record,),
                                                     to_exclude=("TTX",),
                                                     verbose=False, save=False, )
    test_dataset = dpr.make_dataset_from_freq_files(parent_directories=[P.DISK, ], targets_labels=test_targets,
                                                    to_include=("freq_50hz_sample", record,),
                                                    to_exclude=("TTX",),
                                                    verbose=False, save=False, )
    pca, pca_train_dataset = ml.fit_pca(train_dataset, 3)
    ml.plot_pca(pca_train_dataset, 3, show=False, save=True)
    clf = ml.train_RFC_from_dataset(pca_train_dataset)
    pca_test_dataset = ml.apply_pca(pca, test_dataset)

    fused_df = pd.concat([pca_train_dataset, pca_test_dataset], ignore_index=True)
    print(fused_df)
    ml.plot_pca(fused_df, 3, show=False, save=True, commentary="fitted on NI TAHV")


print(datetime.datetime.now())
main()
