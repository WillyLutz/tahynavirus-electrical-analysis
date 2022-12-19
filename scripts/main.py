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
    # todo : pre processing for drugs
    df = dpr.make_dataset_from_freq_files([P.DATA, ], targets_labels=("NI VPA", "NI -DRUG"),
                                          to_include=("T=48H", "freq_50hz"),
                                          to_exclude=("TTX",),
                                          save=False)
    pca, pcdf, _ = ml.fit_pca(df, n_components=2)

    inf_df = dpr.make_dataset_from_freq_files([P.DATA, ], targets_labels=("TAHV -DRUG",),
                                              to_include=("T=48H", "freq_50hz"),
                                              to_exclude=("TTX",),
                                              save=False)
    commentary = "binary drug classification"
    pc_inf = ml.apply_pca(pca, inf_df)
    full_df = pd.concat([pcdf, pc_inf], ignore_index=True)
    ml.plot_pca(full_df, 2, show=False, save=True, commentary=commentary)
    rfc, scores = ml.train_RFC_from_dataset(pcdf)
    ml.test_model(rfc, full_df, training_targets=("NI VPA", "NI -DRUG"),
                  testing_targets=("NI VPA", "NI -DRUG", "TAHV -DRUG"),
                  save=True, show=False, commentary=commentary, iterations=10)
    print(scores)

now = datetime.datetime.now()
print(now)
main()
print("RUN:", datetime.datetime.now()-now)
