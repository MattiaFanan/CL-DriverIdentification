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
# File name: auto_main_derpp.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file executes the hyperparameter search for the DER++ algorithm.

import argparse
import random
import subprocess
import sys

from utils import random_class_order, random_runs_order

# Define the script name and the base arguments
script_name = "main_derpp.py"
base_main_args = [
    "--api_key_file", "api_key.json",
    "--data_path", "data/dataset.csv",
    "--val_ratio", "0.15",
    "--test_ratio", "0.15",
    "--early_stop_patience", "20",
    "--max_epochs", "300",
    "--batch_size", "32",
    "--window_size", "60",
    "--step_size", "6",
    "--wandb_mode", "online",
    "--growing_model",
    "--smoothing_window", "6"
    "--eval_every", "1"
]

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Configure your experiment.')
    # Add arguments
    parser.add_argument('--device', type=str, default='cpu', help='Device to use.')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--api_key_file', type=str, required=True, help='Path to the api key file.')
    parser.add_argument('--wandb_entity', type=str, required=True, help='Wandb entity name.')
    args = parser.parse_args()

    base_main_args = base_main_args + [
        "--api_key_file", str(args.api_key_file),
        "--data_path", str(args.data_path),
        "--wandb_entity", str(args.wandb_entity),
        "--device", args.device,
        "--project", args.project
    ]

    # Define the range of memory sizes to test
    alphas = [0.5, 0.75, 1]
    betas = [0.5, 0.75, 1]
    memory_sizes = [100, 500, 1000, 2000]
    num_different_seeds = 4
    num_different_class_orders = 4
    original_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xexp_list = [1, 2]
    xexp_list_split = [2]

    seeds = [random.randint(0, 1000) for _ in range(num_different_seeds)]

    for xexp in xexp_list[1:]:
        data_orders = [random_class_order(original_classes, xexp) for _ in range(num_different_class_orders)]
        for memory_size in memory_sizes:
            for alpha in alphas:
                for beta in betas:
                    for i, experiences_targets in enumerate(data_orders):
                        for seed in seeds:
                            run_name = f"{xexp}_xexp_size_{memory_size}_alpha_{alpha}_beta_{beta}_seed_{seed}_order_id_{i}"
                            main_args = base_main_args + [
                                "--run_name", run_name,
                                "--seed", str(seed),
                                "--experiences_targets", str(experiences_targets),
                                "--memory_size", str(memory_size),
                                "--alpha", str(alpha),
                                "--beta", str(beta)
                            ]
                            command = [sys.executable, script_name] + main_args
                            print(command)
                            subprocess.run(command)

    for xexp in xexp_list_split:
        data_orders_runs = [random_runs_order(original_classes, xexp) for _ in
                            range(num_different_class_orders)]
        for memory_size in memory_sizes:
            for alpha in alphas:
                for beta in betas:
                    for i, experiences_targets in enumerate(data_orders_runs):
                        for seed in seeds:
                            run_name = f"{xexp}_xexp_split_{memory_size}_size_{alpha}_alpha_{beta}_beta_{seed}_seed_{i}_order_id"
                            main_args = base_main_args + [
                                "--run_name", run_name,
                                "--seed", str(seed),
                                "--experiences_targets", str(experiences_targets),
                                "--memory_size", str(memory_size),
                                "--alpha", str(alpha),
                                "--beta", str(beta)
                            ]
                            command = [sys.executable, script_name] + main_args
                            print(command)
                            subprocess.run(command)

