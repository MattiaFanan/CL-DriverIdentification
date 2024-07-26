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
# File name: cl_dataset.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file contains the implementation of the CLDataset class and the Metric class.

from copy import deepcopy
from typing import Callable, Any, List, Tuple, Union

import numpy
import torch


def get_cldataset_from_ordered_experiences_targets(x, y, ordered_experiences_targets):
    # each cell has a list of indexes of the samples that belong to the corresponding experience
    experiences_datasets_indexes = [
        torch.isin(y, torch.tensor(exp_targets))
        for exp_targets in ordered_experiences_targets]
    Xs = [x[exp_indexes] for exp_indexes in experiences_datasets_indexes]
    Ys = [y[exp_indexes] for exp_indexes in experiences_datasets_indexes]
    return CLDataset([(X, Y) for X, Y in zip(Xs, Ys)])


class CLDataset:

    def __init__(self, data: List[Tuple[Union[numpy.ndarray, torch.Tensor], Union[numpy.ndarray, torch.Tensor]]]):
        for x, y in data:
            assert x.shape[0] == y.shape[0]
        self.data = deepcopy(data)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        return self.data[idx]


class Metric:
    def __init__(
            self,
            name: str,
            metric_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            aggregation_func: Callable[[torch.Tensor], torch.Tensor] = torch.mean):
        self.name = name
        self.metric_func = metric_func
        self.on_update_cls = []
        self.on_aggregate_cls = []
        self.accumulated_result = torch.tensor([]).type(torch.float32)
        self.aggregation_func = aggregation_func

    def register_on_update_callback(self, callback: Callable[[torch.Tensor], Any]):
        self.on_update_cls.append(callback)
        return self

    def register_on_aggregate_callback(self, callback: Callable[[torch.Tensor], Any]):
        self.on_aggregate_cls.append(callback)
        return self

    def update(self, y_pred, y_true, store=True):
        """
        Computes the metric and cals the callbacks with the result.
        If you want, you can store the result in the accumulated_result tensor.
        :param y_pred:
        :param y_true:
        :param store:
        :return:
        """
        y_pred = y_pred.detach().to('cpu')
        y_true = y_true.detach().to('cpu')
        result = self.metric_func(y_pred, y_true)
        # if the result is tensor(0.0), we need to unsqueeze it to make it a tensor([0.0])
        # so that we can concatenate it with other tensors
        if result.dim() == 0:
            result = result.unsqueeze(0)
        if store:
            self.accumulated_result = torch.cat((self.accumulated_result, result))
        for callback in self.on_update_cls:
            callback(result)
        return result

    def aggregate(self):
        """
        Aggregates the results stored in the accumulated_result tensor using the aggregation function you gave.
        :return:
        """
        result = self.aggregation_func(self.accumulated_result)
        for callback in self.on_aggregate_cls:
            callback(result)
        self.accumulated_result = torch.tensor([]).type(torch.float32)
        return result
