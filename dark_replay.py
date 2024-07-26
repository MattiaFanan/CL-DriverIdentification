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
# File name: dark_replay.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file contains the DER++ version of replay.py memory class

import numpy as np


def _sample_per_class(x, y_logits, y_true, n_samples_per_class):
    n_samples_per_class = int(n_samples_per_class)
    classes = np.unique(y_true)
    samples = []
    targets = []
    logits = []
    for c in classes:
        c_mask = np.where(y_true == c)[0]
        c_sample_indexes = np.random.choice(c_mask, min(n_samples_per_class, len(c_mask)), replace=False)
        samples.append(x[c_sample_indexes])
        targets.append(y_true[c_sample_indexes])
        logits.append(y_logits[c_sample_indexes])
    return np.concatenate(samples), np.concatenate(logits), np.concatenate(targets)


class DarkReplayMemory:
    def __init__(self, mem_size, x_dtype=np.float32, y_true_dtype=np.int64):
        # Initialize any custom attributes or methods here
        self.mem_size = mem_size
        self._class_encounters = {}
        self._x_dtype = x_dtype
        self._y_true_dtype = y_true_dtype

        self.x = np.array([], dtype=self._x_dtype)
        self.y_logits = np.array([], dtype="object")
        self.y_true = np.array([], dtype=self._y_true_dtype)

    @property
    def seen_classes(self):
        return [k for k in self._class_encounters.keys()]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_logits[idx], self.y_true[idx]

    @property
    def is_empty(self):
        return len(self.x) == 0

    @staticmethod
    def __safe_concatenate(a, b):
        if not a.size:
            return b
        if not b.size:
            return a
        return np.concatenate((a, b))

    def _add_instances_from_new_classes(self, x, y_logits, y_true):
        if not y_true.size:
            return

        # Sample the same number of instances in class to keep the memory balanced
        num_of_new_classes = len(np.unique(y_true))
        ideal_num_points_per_class = int(self.mem_size / (num_of_new_classes + len(self.seen_classes)))
        x, y_logits, y_true = _sample_per_class(x, y_logits, y_true, ideal_num_points_per_class)

        # -#-#-#-#-# if there is enough free memory, add the new samples
        free_space = self.mem_size - len(self.x)
        if len(x) <= free_space:
            # use the numpy hstack to add the new samples in the last dimension
            # concatenate gives an error if one of the arrays is empty
            self.x = self.__safe_concatenate(self.x, x)
            self.y_logits = self.__safe_concatenate(self.y_logits, y_logits)
            self.y_true = self.__safe_concatenate(self.y_true, y_true)
            return

        # -#-#-#-#-# if we got here, we need to free some space
        for class_key in self.seen_classes:
            class_indices = np.where(self.y_true == class_key)[0]
            num_samples_to_remove = len(class_indices) - ideal_num_points_per_class
            if num_samples_to_remove > 0:
                # remove the oldest samples
                random_indices = np.random.choice(class_indices, num_samples_to_remove, replace=False)
                self.x = np.delete(self.x, random_indices, axis=0)
                self.y_logits = np.delete(self.y_logits, random_indices, axis=0)
                self.y_true = np.delete(self.y_true, random_indices, axis=0)

        # -#-#-#-#-# add new ones
        self.x = self.__safe_concatenate(self.x, x)
        self.y_logits = self.__safe_concatenate(self.y_logits, y_logits)
        self.y_true = self.__safe_concatenate(self.y_true, y_true)

    def _update_class_encounters(self, y):
        for c in np.unique(y):
            self._class_encounters[c] = self._class_encounters.get(c, 0) + 1

    def _resample_instances_from_old_classes(self, x, y_logits, y_true):
        # Calculate the number of samples to resample for each old class
        resample_ratios = {
            class_key: 1 / (self._class_encounters[class_key] + 1)
            for class_key in np.unique(self.y_true)
            if class_key in self._class_encounters}

        # Step 3: Resample for each old class
        for class_key, ratio in resample_ratios.items():
            # Find the indices of instances belonging to the current class
            class_indices_in_memory = np.where(self.y_true == class_key)[0]
            class_indices_in_new_data = np.where(y_true == class_key)[0]

            # Calculate the number of samples to resample
            num_samples_to_replace = int(len(class_indices_in_memory) * ratio)
            num_samples_to_replace = min(num_samples_to_replace, len(class_indices_in_new_data))

            # Resample indices
            resampled_indices = np.random.choice(class_indices_in_memory, num_samples_to_replace, replace=False)
            sampled_new_indices = np.random.choice(class_indices_in_new_data, num_samples_to_replace, replace=False)

            # Add the resampled indices to the list of indices to replace
            self.x[resampled_indices] = x[sampled_new_indices]
            self.y_logits[resampled_indices] = y_logits[sampled_new_indices]
            self.y_true[resampled_indices] = y_true[sampled_new_indices]

    def add_to_memory(self, x, y_logits, y_true, resample_old_classes=True):

        old_mask = np.isin(y_true, self.seen_classes)
        # new classes change the memory repartition between classes, because we want a balanced memory
        new_x = x[~old_mask]
        new_y_logits = y_logits[~old_mask]
        new_y_true = y_true[~old_mask]
        self._add_instances_from_new_classes(new_x, new_y_logits, new_y_true)
        if resample_old_classes:
            # instances from old classes will be mixed half-and-half with the ones already in memory after repartition
            old_x = x[old_mask]
            old_y_logits = y_logits[old_mask]
            old_y_true = y_true[old_mask]
            self._resample_instances_from_old_classes(old_x, old_y_logits, old_y_true)
        # update the class encounters for each class in the new y
        self._update_class_encounters(y_true)

    def sample(self, n_samples):
        # if empty memory return empty tensors
        if self.is_empty:
            return (
                np.array([], dtype=self._x_dtype),
                [],
                np.array([], dtype=self._y_true_dtype))
        sampled_indices = np.random.choice(len(self.x), n_samples, replace=False)
        return self.x[sampled_indices], self.y_logits[sampled_indices], self.y_true[sampled_indices]
