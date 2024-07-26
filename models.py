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
# File name: models.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file contains the implementation of the LSTM model


import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(num_features, 128, batch_first=True)
        self.dp1 = nn.Dropout(0.5)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.dp2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
        #self.softmax = nn.Softmax(dim=-1)
        self.final = nn.Sigmoid()

    def forward(self, x):
        # Flatten the parameters of the LSTM, otherwise it will throw a warning
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        # LSTM returns output_sequence, (hidden_states, cell_states)
        # if batch_first=True, the output is (batch_size, seq_len, num_directions * hidden_size)
        # if batch_first=False, the output is (seq_len, batch_size, num_directions * hidden_size)
        output1, _ = self.lstm1(x)
        output1 = self.dp1(output1)
        output2, _ = self.lstm2(output1)
        # Take the output from the last timestep of the sequence
        output_last_timestep = output2[:, -1, :]
        output_last_timestep = self.dp2(output_last_timestep)
        # Pass through the fully connected layer
        output_fc = self.fc(output_last_timestep)
        # Apply sigmoid activation
        output_final = self.final(output_fc)

        return output_final

    def resize_output_layer(self, num_classes):
        new_fc = nn.Linear(128, num_classes)
        to_copy = min(self.fc.weight.shape[0], new_fc.weight.shape[0])
        # Copy the weights and biases from the old layer to the new layer
        if to_copy > 0:
            with torch.no_grad():
                new_fc.weight.data[:to_copy, :] = self.fc.weight.data
                new_fc.bias.data[:to_copy] = self.fc.bias.data
        self.fc = new_fc.to(self.fc.weight.device)
