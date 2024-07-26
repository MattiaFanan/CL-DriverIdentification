# Copyright 2024 Mattia Fanan
# Copyright 2024 Davide Dalle Pezze
# Copyright 2024 Emad Efatinasab
# Copyright 2024 Ruggero Carli
# Copyright 2024 Mirco Rampazzo
# Copyright 2024 Gian Antonio Susto
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File name: test_extraction.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file contains the functions to extract test, train and validation data from the dataset

import numpy as np
import pandas as pd


def sliding_window(data, window_size, step_size):
    # Create a tensor with the windows (add a dimension to the data to make it 3D [n_windows, window_size, n_features])
    return np.stack([data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step_size)])


def make_train_test_val_windows(
        x, y, window_size: int, step_size: int,
        val_ratio: float, test_ratio: float):
    if np.unique(y).shape[0] != 1:
        raise ValueError("There can be only one class in the run")
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    label = y[0]

    windows = sliding_window(x, window_size, step_size)
    index_permutation = np.random.permutation(windows.shape[0])

    val_size = int(val_ratio * windows.shape[0])
    test_size = int(test_ratio * windows.shape[0])

    sliding_val_x = windows[index_permutation[:val_size]]
    sliding_test_x = windows[index_permutation[val_size:val_size+test_size]]
    sliding_train_x = windows[index_permutation[val_size+test_size:]]

    sliding_val_y = np.array([label] * sliding_val_x.shape[0])
    sliding_test_y = np.array([label] * sliding_test_x.shape[0])
    sliding_train_y = np.array([label] * sliding_train_x.shape[0])

    return sliding_train_x, sliding_train_y, sliding_test_x, sliding_test_y, sliding_val_x, sliding_val_y


def get_test_train_val_windowed_runs(df, run_column_name, y_column_name, window_size, step_size, val_ratio, test_ratio):
    runs = {
        label: {
            run: df[(df[y_column_name] == label) & (df[run_column_name] == run)]
            for run in df[df[y_column_name] == label][run_column_name].unique()
        }
        for label in df[y_column_name].unique()
    }

    runs_xy = {
        label: {
            run: (data.drop([y_column_name, run_column_name], axis=1), data[y_column_name])
            for run, data in label_dict.items()
        }
        for label, label_dict in runs.items()
    }

    train_windows = {}
    val_windows = {}
    test_windows = {}

    for label, label_dict in runs_xy.items():
        for run, data in label_dict.items():
            x, y = data
            tr_x, tr_y, te_x, te_y, val_x, val_y = make_train_test_val_windows(x, y,
                                                                               window_size, step_size,
                                                                               val_ratio, test_ratio)
            if tr_x.shape[0] == 0:
                raise ValueError("found empty training set")
            if val_x.shape[0] == 0:
                val_windows = None
            if te_x.shape[0] == 0:
                test_windows = None

            if label not in train_windows:
                train_windows[label] = {}
            train_windows[label][run] = (tr_x, tr_y)
            if val_windows is not None:
                if label not in val_windows:
                    val_windows[label] = {}
                val_windows[label][run] = (val_x, val_y)
            if test_windows is not None:
                if label not in test_windows:
                    test_windows[label] = {}
                test_windows[label][run] = (te_x, te_y)

    return train_windows, test_windows, val_windows
