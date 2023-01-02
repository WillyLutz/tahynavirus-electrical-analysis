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
from scipy.spatial import ConvexHull, convex_hull_plot_2d

pd.set_option('display.max_columns', None)
import complete_procedures as cp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    for tested in ['VPA', 'CPZ', 'MTCL', 'FXT']:
        train_df = dpr.make_dataset_from_freq_files([P.DATA, ], targets_labels=(f"NI {tested}", "NI -DRUG"),
                                                    to_include=("T=48H", "freq_50hz"),
                                                    to_exclude=("TTX", "NI/4", "TAHV/4"),
                                                    save=False,
                                                    separate_organoids=False)
        pca, pcdf, _ = ml.fit_pca(train_df, n_components=3)
        test_df = dpr.make_dataset_from_freq_files([P.DATA, ], targets_labels=(f"TAHV -DRUG",),
                                                   to_include=("T=48H", "freq_50hz"),
                                                   to_exclude=("TTX", "NI/4", "TAHV/4"),
                                                   save=False,
                                                   separate_organoids=False)
        pc_inf = ml.apply_pca(pca, test_df)
        full_df = pd.concat([pcdf, pc_inf], ignore_index=True)

        ml.plot_pca(full_df, n_components=3, show=False, save=False, commentary="3D binary testing on drugs no outsider",
                    points=True, metrics=False)
        rfc, _ = ml.train_RFC_from_dataset(pcdf)

        ml.test_model(rfc, full_df, training_targets=("NI -DRUG", f"NI {tested}"), testing_targets=("NI -DRUG", f"NI {tested}", 'TAHV -DRUG',), show=False, save=True, commentary="3D binary testing on drugs no outsider")


now = datetime.datetime.now()
print(now)
main()
print("RUN:", datetime.datetime.now() - now)
