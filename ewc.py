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
# File name: ewc.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file contains the EWC class, which is used to implement the Elastic Weight Consolidation algorithm.


import torch
from torch import nn


def _get_fisher_diag(model, dataset, criterion, empirical=True):
    fisher = {}
    # train is required to run backpropagation
    model.train()
    # dropout is a problem here because randomly goes on and off
    # remove dropouts
    param_dropout = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            param_dropout[name] = module.p
            module.p = 0
        elif isinstance(module, nn.LSTM):
            param_dropout[name] = module.dropout
            module.dropout = 0
        elif isinstance(module, nn.GRU):
            param_dropout[name] = module.dropout
            module.dropout = 0
    model.zero_grad()

    for x, y in dataset:
        model.zero_grad()
        output = model(x)
        if empirical:
            label = y
        else:
            label = torch.argmax(output, dim=1)
        loss = criterion(output, label)
        loss.backward()

        for n, p in model.named_parameters():
            if n not in fisher:
                fisher[n] = torch.zeros_like(p.grad).to(p.device)
            g = p.grad.clone().detach()
            fisher[n].data += g.pow(2) / len(dataset.dataset)  # Normalize by total dataset size

    # reset model gradients
    model.zero_grad()
    # reset dropouts
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = param_dropout[name]
        elif isinstance(module, nn.LSTM):
            module.dropout = param_dropout[name]
        elif isinstance(module, nn.GRU):
            module.dropout = param_dropout[name]

    return fisher


class EWC:
    def __init__(self, ewc_lambda, empirical=True):
        self.ewc_lambda = ewc_lambda
        self.empirical = empirical
        self.saved_params = []
        self.saved_importance = []

    def update_importance(self, model, dataset, criterion):
        self.saved_params.append({n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad})
        self.saved_importance.append(_get_fisher_diag(model, dataset, criterion, self.empirical))

    def get_ewc_loss(self, model):
        if len(self.saved_params) <= 0:
            return 0
        loss = 0
        for exp in range(len(self.saved_params)):
            for name, p in model.named_parameters():
                if p.requires_grad is False:
                    continue
                if name not in self.saved_params[exp]:
                    continue
                old_params = self.saved_params[exp][name]
                old_dim = old_params.shape[0]
                sliced_p = p[:old_dim, ...]
                distance = (sliced_p - old_params).pow(2)
                delta_loss = self.saved_importance[exp][name] * distance
                loss += delta_loss.sum()

        return self.ewc_lambda * loss

