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
    timepoint = "T=48H"
    min_freq = 0
    max_freq = 5000
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    cp.Confusion_matrix(0, 5000, "batch 1", "T=48H") # Exemple of use

now = datetime.datetime.now()
print(now)
main()

print("RUN:", datetime.datetime.now() - now)
