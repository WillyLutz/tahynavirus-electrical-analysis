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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    thdf, _ = ml.principal_component_analysis(("NI", "TAHV", "NI+RG27", "TAHV+RG27"), 3, save=False, show=False)
    print(thdf.columns)
    # todo pca into RFC or unsupervised algorithm




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
