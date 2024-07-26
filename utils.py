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
# File name: utils.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file contains utility functions to generate the sequence of the classes and runs for the experiments


import random


def random_class_order(original_classes, class_per_exp):
    # Create a copy of the original classes
    classes = original_classes.copy()
    # Shuffle the classes
    random.shuffle(classes)

    if class_per_exp == 1:
        return [classes[:2]] + [[c] for c in classes[2:]]
    else:
        return [classes[i:i + class_per_exp] for i in range(0, len(classes), class_per_exp)]


def random_runs_order(original_classes, data_per_exp):
    # Create a copy of the original classes
    classes = original_classes.copy()
    runs = [1, 2]
    order = [(c, run) for c in classes for run in runs]
    # Shuffle the classes
    random.shuffle(order)
    # make sure the first two elements are from different classes
    selected_classes = random.sample(classes, 2)
    selected_runs = [(c, random.choice(runs)) for c in selected_classes]

    index_selected_runs = [order.index((c, r)) for c, r in selected_runs]
    tmp = order[0]
    order[0] = order[index_selected_runs[0]]
    order[index_selected_runs[0]] = tmp
    tmp = order[1]
    order[1] = order[index_selected_runs[1]]
    order[index_selected_runs[1]] = tmp

    if data_per_exp == 1:
        return [order[:2]] + [[r] for r in order[2:]]
    else:
        return [order[i:i + data_per_exp] for i in range(0, len(order), data_per_exp)]