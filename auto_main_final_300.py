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
# File name: auto_main_final_300.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This file executes all the experiments for all the 3 scenarios:
# 1) 2 classes per experience
# 2) 2 classes for the first experience and 1 class for the following experiences
# 3) 2 runs per experience (a run is a drive session for a specific driver)
# The experiments are executed with different seeds and different data orders in the experiences.

import argparse
import random
import subprocess
import sys

from utils import random_class_order, random_runs_order

# Define the script name and the base arguments
base_main_args = [
    "--api_key_file", "api_key.json",
    "--data_path", "data/dataset.csv",
    "--val_ratio", "0",
    "--test_ratio", "0.3",
    "--early_stop_patience", "20",
    "--max_epochs", "5",
    "--batch_size", "32",
    "--window_size", "60",
    "--step_size", "6",
    "--wandb_mode", "online",
    "--growing_model",
    "--eval_every", "1",
    "--smoothing_window", "6"

]
base_script_name = "main_baseline.py"
replay_script_name = "main_replay.py"
ewc_script_name = "main_ewc.py"
lwf_script_name = "main_lwf.py"
derpp_script_name = "main_derpp.py"
cumulative_script_name = "main_cumulative.py"

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
    num_different_seeds = 4
    num_different_class_orders = 4
    original_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    xexp_list = [1, 2]
    xexp_list_split = [2]

    seeds = [random.randint(0, 1000) for _ in range(num_different_seeds)]

    for xexp in xexp_list:
        data_orders = [random_class_order(original_classes, xexp) for _ in range(num_different_class_orders)]
        # fine tuning
        for i, experiences_targets in enumerate(data_orders):
            for seed in seeds:
                run_name = f"finetuning_{xexp}_xexp_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--group_name", "finetuning",
                    "--wandb_folder", f"./wandb_xexp_{xexp}"
                ]
                command = [sys.executable, base_script_name] + main_args
                print(command)
                subprocess.run(command)
        # replay
        for i, experiences_targets in enumerate(data_orders):
            for seed in seeds:
                run_name = f"replay_{xexp}_xexp_1000_size_0.5_mixin_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--memory_size", str(1000),
                    "--memory_mixin", str(0.5),
                    "--group_name", "replay",
                    "--wandb_folder", f"./wandb_xexp_{xexp}"
                ]
                command = [sys.executable, replay_script_name] + main_args
                print(command)
                subprocess.run(command)
        # EWC
        for i, experiences_targets in enumerate(data_orders):
            for seed in seeds:
                run_name = f"ewc_{xexp}_xexp_10000_lambda_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--ewc_lambda", str(10000),
                    "--group_name", "ewc",
                    "--wandb_folder", f"./wandb_xexp_{xexp}"
                ]
                command = [sys.executable, ewc_script_name] + main_args
                print(command)
                subprocess.run(command)
        # LWF
        for i, experiences_targets in enumerate(data_orders):
            for seed in seeds:
                run_name = f"lwf_{xexp}_xexp_5_lambda_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--lwf_lambda", str(5),
                    "--group_name", "lwf",
                    "--wandb_folder", f"./wandb_xexp_{xexp}"
                ]
                command = [sys.executable, lwf_script_name] + main_args
                print(command)
                subprocess.run(command)
        # DERPP
        for i, experiences_targets in enumerate(data_orders):
            for seed in seeds:
                run_name = f"derpp_{xexp}_xexp_1_alpha_1_beta_1000_size_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--alpha", str(1),
                    "--beta", str(1),
                    "--memory_size", str(1000),
                    "--group_name", "derpp",
                    "--wandb_folder", f"./wandb_xexp_{xexp}"
                ]
                command = [sys.executable, derpp_script_name] + main_args
                print(command)
                subprocess.run(command)
        #    ########################## cumulative
        for i, experiences_targets in enumerate(data_orders):
            for seed in seeds:
                run_name = f"cumulative_co_{xexp}_xexp_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--group_name", "cumulative_co",
                    "--wandb_folder", f"./wandb_xexp_{xexp}"
                ]
                command = [sys.executable, cumulative_script_name] + main_args
                print(command)
                subprocess.run(command)
    #
    ########################## with runs #####################################
    for xexp_split in xexp_list_split:
        data_orders_drive = [random_runs_order(original_classes, xexp_split) for _ in range(num_different_class_orders)]
        # fine tuning
        for i, experiences_targets in enumerate(data_orders_drive):
            for seed in seeds:
                run_name = f"finetuning_{xexp_split}_xexp_split_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--group_name", "finetuning",
                    "--wandb_folder", f"./wandb_xexp_{xexp_split}_split"
                ]
                command = [sys.executable, base_script_name] + main_args
                print(command)
                subprocess.run(command)
        # replay
        for i, experiences_targets in enumerate(data_orders_drive):
            for seed in seeds:
                run_name = f"replay_{xexp_split}_xexp_split_1000_size_0.5_mixin_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--memory_size", str(1000),
                    "--memory_mixin", str(0.5),
                    "--group_name", "replay",
                    "--wandb_folder", f"./wandb_xexp_{xexp_split}_split"
                ]
                command = [sys.executable, replay_script_name] + main_args
                print(command)
                subprocess.run(command)
        # EWC
        for i, experiences_targets in enumerate(data_orders_drive):
            for seed in seeds:
                run_name = f"ewc_{xexp_split}_xexp_split_10000_lambda_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--ewc_lambda", str(10000),
                    "--group_name", "ewc",
                    "--wandb_folder", f"./wandb_xexp_{xexp_split}_split"
                ]
                command = [sys.executable, ewc_script_name] + main_args
                print(command)
                subprocess.run(command)
        # LWF
        for i, experiences_targets in enumerate(data_orders_drive):
            for seed in seeds:
                run_name = f"lwf_{xexp_split}_xexp_split_5_lambda_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--lwf_lambda", str(5),
                    "--group_name", "lwf",
                    "--wandb_folder", f"./wandb_xexp_{xexp_split}_split"
                ]
                command = [sys.executable, lwf_script_name] + main_args
                print(command)
                subprocess.run(command)
        # DERPP
        for i, experiences_targets in enumerate(data_orders_drive):
            for seed in seeds:
                run_name = f"derpp_{xexp_split}_xexp_split_1_alpha_1_beta_1000_size_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--alpha", str(1),
                    "--beta", str(1),
                    "--memory_size", str(1000),
                    "--group_name", "derpp",
                    "--wandb_folder", f"./wandb_xexp_{xexp_split}_split"
                ]
                command = [sys.executable, derpp_script_name] + main_args
                print(command)
                subprocess.run(command)
        #    ########################## cumulative
        for i, experiences_targets in enumerate(data_orders_drive):
            for seed in seeds:
                run_name = f"cumulative_co_{xexp_split}_xexp_split_{seed}_seed_{i}_order_id"
                main_args = base_main_args + [
                    "--run_name", run_name,
                    "--seed", str(seed),
                    "--experiences_targets", str(experiences_targets),
                    "--group_name", "cumulative_co",
                    "--wandb_folder", f"./wandb_xexp_{xexp_split}_split"
                ]
                command = [sys.executable, cumulative_script_name] + main_args
                print(command)
                subprocess.run(command)

    # ##############################MULTI-CASS##############################
    # all in once

    for seed in [random.randint(0, 1000) for _ in range(16)]:
        run_name = f"multi_seed_{seed}"
        main_args = base_main_args + [
            "--run_name", run_name,
            "--seed", str(seed),
            "--experiences_targets", str([original_classes]),
            "--group_name", "multi",
            "--wandb_folder", f"./wandb_multi"
        ]
        command = [sys.executable, base_script_name] + main_args
        print(command)
        subprocess.run(command)

