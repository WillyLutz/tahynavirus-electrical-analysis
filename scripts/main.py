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
import pickle
import forestci as fci
from pathlib import Path
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff
import fiiireflyyy.fireprocess as fp
from scipy.stats.stats import pearsonr

pd.set_option('display.max_columns', None)
import complete_procedures as cp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks


def main():
    class1_df = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                                to_include=("odd_harmonics", timepoint),
                                                to_exclude=("TTX", "RG27", "flattened", "TAHV"),
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_samples=batches["batch 4"],
                                                separate_samples=False,
                                                label_comment=f"",
                                                target_keys={'NI': 'Mock',
                                                             'TAHV': 'TAHV'})  # IMPORTANT, else it may result in an empty dataframe
    discarded_class1 = fp.discard_outliers_by_iqr(class1_df, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    percentiles = 0.1
    base = "/home/wlutz/PycharmProjects/tahynavirus-electrical-analysis/datasets"
    t0mock = pd.read_csv(f'{base}/DATASET_T=0 MOCK.csv')
    dt0mock = fp.discard_outliers_by_iqr(t0mock, low_percentile=percentiles, high_percentile=1 - percentiles,
                                         mode='capping')
    dt0mock['label'].replace('Mock', 'Mock\n0min', inplace=True)

    t0vlpp = pd.read_csv(f'{base}/DATASET_T=0 VLP+.csv')
    dt0vlpp = fp.discard_outliers_by_iqr(t0vlpp, low_percentile=percentiles, high_percentile=1 - percentiles,
                                         mode='capping')
    dt0vlpp['label'].replace('VLP+', 'VLP+\n0min', inplace=True)

    t24mock = pd.read_csv(f'{base}/DATASET_T=24H MOCK.csv')
    dt24mock = fp.discard_outliers_by_iqr(t24mock, low_percentile=percentiles, high_percentile=1 - percentiles,
                                          mode='capping')
    dt24mock['label'].replace('Mock', 'Mock\n24h', inplace=True)

    t0spike = pd.read_csv(f'{base}/DATASET_T=0 SPIKE.csv')
    dt0spike = fp.discard_outliers_by_iqr(t0spike, low_percentile=percentiles, high_percentile=1 - percentiles,
                                          mode='capping')
    dt0spike['label'].replace('Spike', 'Spike\n0min', inplace=True)

    t0inf = pd.read_csv(f'{base}/DATASET_T=0 INF.csv')
    dt0inf = fp.discard_outliers_by_iqr(t0inf, low_percentile=percentiles, high_percentile=1 - percentiles,
                                        mode='capping')
    dt0inf['label'].replace('Sars-CoV', 'Sars-CoV\n0min', inplace=True)

    t24inf = pd.read_csv(f'{base}/DATASET_T=24H INF.csv')
    dt24inf = fp.discard_outliers_by_iqr(t24inf, low_percentile=percentiles, high_percentile=1 - percentiles,
                                         mode='capping')
    dt24inf['label'].replace('Sars-CoV', 'Sars-CoV\n24h', inplace=True)

    rfc, _ = fl.train_RFC_from_dataset(pd.concat([dt0mock,
                                                  dt0inf,
                                                  dt0vlpp,
                                                  dt0spike,
                                                  dt24mock,
                                                  dt24inf], ignore_index=True), )

    inf = pd.read_csv(f'{base}/DATASET_T=24H INF.csv')
    dinf = fp.discard_outliers_by_iqr(inf, low_percentile=percentiles, high_percentile=1 - percentiles,
                                      mode='capping')
    dinf['label'].replace('Sars-CoV', 'Sars-CoV\n24h', inplace=True)

    vlpp = pd.read_csv(f'{base}/DATASET_T=24H VLP+.csv')
    dvlpp = fp.discard_outliers_by_iqr(vlpp, low_percentile=percentiles, high_percentile=1 - percentiles,
                                       mode='capping')
    dvlpp['label'].replace('VLP+', 'VLP+\n24h', inplace=True)

    spike = pd.read_csv(f'{base}/DATASET_T=24H SPIKE.csv')
    dspike = fp.discard_outliers_by_iqr(spike, low_percentile=percentiles, high_percentile=1 - percentiles,
                                        mode='capping')
    dspike['label'].replace('Spike', 'Spike\n24h', inplace=True)

    testdf = pd.concat([dvlpp, dinf, dspike, ], ignore_index=True)

    # globaldf = pd.concat([traindf, testdf], ignore_index=True)
    globaldf = pd.concat([dt0mock,
                          dt0inf,
                          dt0vlpp,
                          dt0spike,
                          dt24mock,
                          dinf,
                          dspike,
                          dvlpp, ], ignore_index=True)

    fl.test_rfc_by_confusion(rfc, globaldf, training_targets=('Mock\n0min',
                                                              'Sars-CoV\n0min',
                                                              'VLP+\n0min',
                                                              'Spike\n0min',
                                                              'Mock\n24h',
                                                              'Sars-CoV\n24h'),
                             testing_targets=tuple(set(list(('Mock\n0min',
                                                             'Sars-CoV\n0min',
                                                             'VLP+\n0min',
                                                             'Spike\n0min',
                                                             'Mock\n24h',
                                                             'Sars-CoV\n24h',
                                                             'VLP+\n24h',
                                                             'Spike\n24h')))),
                             show=True, verbose=False, savepath='',
                             title=f"",
                             iterations=5, )

    # training(mock0, vlp-0, vlp+0, spike0, inf0, mock24 as MOCK and inf24 as INF) testing()


now = datetime.datetime.now()
print(now)
main()

print("RUN:", datetime.datetime.now() - now)
