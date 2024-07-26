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
# File name: main_derpp.py
# Project: CL-DriverIdentification
# Date (format yyyy-mm-dd): 2024-07-26
# Authors: Mattia Fanan, Davide Dalle Pezze, Emad Efatinasab, Ruggero Carli, Mirco Rampazzo, Gian Antonio Susto
# Description: This script is used to run the entire experiment of training and test
# using the Dark Experience Replay (DER++) strategy.

import argparse
import ast
import json
import os
import random
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn import preprocessing
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from cl_dataset import Metric
from dark_replay import DarkReplayMemory
from prediction_smoothing import moving_average_by_column
from models import LSTM
from test_extraction import get_test_train_val_windowed_runs

class_column = "Class"
run_column = "PathOrder"
features_to_drop = [
    'Time(s)',
    'Filtered_Accelerator_Pedal_value',
    'Inhibition_of_engine_fuel_cut_off',
    'Fuel_Pressure',
    'Torque_scaling_factor(standardization)',
    'Glow_plug_control_request']


def map_dict(dictionary, mapping):
    return {
        mapping[label]: {
            run: (data[0], map_y(data[1], mapping)) for run, data in label_dict.items()
        }
        for label, label_dict in dictionary.items()
    }


def derpp_train_eval_model(
        model, train_dict, val_dict, test_dict, criterion, optimizer,
        experiences_targets, alpha, beta, memory_size, batch_size,
        n_epochs, smoothing_window, early_stop_patience=np.inf,
        early_stop_margin=0, eval_every=1,
        device="cpu", growing_model=False):
    model = model.to(device)
    criterion = criterion.to(device)

    n_experiences = len(experiences_targets)
    replay_memory = DarkReplayMemory(mem_size=memory_size)
    wandb.define_metric("Train/*", step_metric="Epoch")
    wandb.define_metric("Val/*", step_metric="Epoch")
    wandb.define_metric("Test/*", step_metric="Experience")
    wandb.define_metric("Training Time", step_metric="Experience")
    wandb.define_metric("Test Cumulative/*", step_metric="Experience")
    train_metrics = [
        Metric("Train/Accuracy", accuracy).register_on_aggregate_callback(
            lambda result: wandb.log({"Train/Accuracy": result})
        ),
        Metric("Train/Loss", cross_entropy).register_on_aggregate_callback(
            lambda result: wandb.log({"Train/Loss": result})
        )
    ]
    val_metrics = [
        Metric("Val/Accuracy", accuracy).register_on_update_callback(
            lambda result: wandb.log({"Val/Accuracy": result})
        ),
        Metric("Val/Loss", cross_entropy).register_on_update_callback(
            lambda result: wandb.log({"Val/Loss": result})
        )
    ]
    cumulative_test_metrics = [
        Metric("Test Cumulative/Accuracy", accuracy).register_on_update_callback(
            lambda result: wandb.log({"Test Cumulative/Accuracy": result})
        ),
        Metric("Test Cumulative/Loss", cross_entropy).register_on_update_callback(
            lambda result: wandb.log({"Test Cumulative/Loss": result})
        )
    ]
    # label=label is needed since f"" is evaluated at runtime and label would otherwise be the last value of the loop
    test_metrics = [
        Metric(f"Test/Accuracy/class_{label}", accuracy).register_on_update_callback(
            lambda result, label=label: wandb.log({f"Test/Accuracy/class_{label}": result})
        )
        for label in train_dict.keys()
    ]
    test_metrics += [
        Metric(f"Test/Loss/class_{label}", cross_entropy).register_on_update_callback(
            lambda result, label=label: wandb.log({f"Test/Loss/class_{label}": result})
        )
        for label in train_dict.keys()
    ]

    test_forgetting_metrics = [
        Metric(f"Test/Forgetting/class_{label}", forgetting()).register_on_update_callback(
            lambda result, label=label: wandb.log({f"Test/Forgetting/class_{label}": result})
        )
        for label in train_dict.keys()
    ]

    wandb.define_metric("Smoothing Time", step_metric="Experience")
    wandb.define_metric("Test Smoothened/*", step_metric="Experience")
    smoothened_test_metrics = [
        Metric("Test Smoothened/Accuracy", accuracy).register_on_update_callback(
            lambda result: wandb.log({"Test Smoothened/Accuracy": result})
        ),
        Metric("Test Smoothened/Loss", cross_entropy).register_on_update_callback(
            lambda result: wandb.log({"Test Smoothened/Loss": result})
        )
    ]

    # remap the classes to the order specified in the config
    mapping, de_mapping = get_class_mapper(experiences_targets)
    r_train_dict = map_dict(train_dict, mapping)
    if val_dict is not None:
        r_val_dict = map_dict(val_dict, mapping)
    r_test_dict = map_dict(test_dict, mapping)
    r_experiences_targets = [
        [
            (mapping[target[0]], target[1]) if isinstance(target, tuple) else mapping[target]
            for target in exp
        ]
        for exp in experiences_targets
    ]

    total_epochs = 0
    classes_seen_so_far = set()
    smoothed_pred_accumulator = []
    pred_accumulator = []

    for current_experience in range(n_experiences):
        exp_time = 0
        start_time = time.perf_counter()
        exp_targets = r_experiences_targets[current_experience]
        if isinstance(exp_targets[0], tuple):
            train_exp_x, train_exp_y = zip(*[r_train_dict[label][run] for label, run in exp_targets])
        else:
            train_exp_x, train_exp_y = zip(*[data for label in exp_targets for data in r_train_dict[label].values()])
        train_exp_x = torch.cat([torch.from_numpy(array) for array in train_exp_x]).to(device).float()
        train_exp_y = torch.cat([torch.from_numpy(array) for array in train_exp_y]).to(device).long()
        exp_time += time.perf_counter() - start_time

        wandb.log({"Experience": current_experience + 1})
        print(f"Experience {current_experience + 1}/{n_experiences}")
        new_classes = set(torch.unique(train_exp_y).tolist()).difference(classes_seen_so_far)
        print(f"Classes seen so far: {[de_mapping[label] for label in classes_seen_so_far]}")
        print(
            f"Current experience targets: {[(de_mapping[t[0]], t[1]) if isinstance(t, tuple) else de_mapping[t] for t in exp_targets]}")
        print(f"New classes: {[de_mapping[label] for label in new_classes]}")

        start_time = time.perf_counter()
        if growing_model and len(new_classes) > 0:
            model.resize_output_layer(len(classes_seen_so_far) + len(new_classes))

        experience_train_dataloader = DataLoader(
            TensorDataset(train_exp_x, train_exp_y),
            batch_size=batch_size, shuffle=True, drop_last=True)
        exp_time += time.perf_counter() - start_time

        patience_counter = 0
        best_loss_so_far = np.inf
        since_last_eval = 0

        if val_dict is not None:
            # validation data on the current experience
            if isinstance(exp_targets[0], tuple):
                val_exp_x, val_exp_y = zip(*[r_val_dict[label][run] for label, run in exp_targets])
            else:
                val_exp_x, val_exp_y = zip(*[data for label in exp_targets for data in r_val_dict[label].values()])
            val_exp_x = torch.cat([torch.from_numpy(array) for array in val_exp_x]).to(device).float()
            val_exp_y = torch.cat([torch.from_numpy(array) for array in val_exp_y]).to(device).long()

        for epoch in tqdm(range(n_epochs), unit="epoch", total=n_epochs):
            start_time = time.perf_counter()
            if val_dict is not None:
                if patience_counter >= early_stop_patience:
                    break
                since_last_eval += 1
            total_epochs += 1
            wandb.log({"Epoch": total_epochs})
            # train the model
            model.train()  # Set the model to training mode
            for train_x, train_y in experience_train_dataloader:

                train_x = train_x.to(device)
                train_y = train_y.to(device)

                optimizer.zero_grad()
                train_outputs = model(train_x)
                loss = criterion(train_outputs, train_y)

                x_logits_memory, y_logits_memory, _ = replay_memory.sample(batch_size)
                if len(x_logits_memory) > 0:
                    x_logits_memory = torch.tensor(x_logits_memory).to(device).float()
                    logits_memory_outputs = model(x_logits_memory)

                    # copy and rewrite with y_logits_memory, so that missing dimensions wil not contribute to mse
                    coherent_y_logits_memory = logits_memory_outputs.detach().clone()
                    for i in range(len(y_logits_memory)):
                        sub_tensor = torch.tensor(y_logits_memory[i]).to(device).float()
                        coherent_y_logits_memory[i, :len(sub_tensor)] = sub_tensor
                    coherent_y_logits_memory = coherent_y_logits_memory.to(device).float()

                    loss_logits = alpha * nn.MSELoss()(logits_memory_outputs, coherent_y_logits_memory)
                    wandb.log({"Train/Loss_Logits": loss_logits})
                    loss += loss_logits

                x_true_memory, _, y_true_memory = replay_memory.sample(batch_size)
                if len(x_true_memory) > 0:
                    x_true_memory = torch.tensor(x_true_memory).to(device).float()
                    y_true_memory = torch.tensor(y_true_memory).to(device).long()
                    true_memory_outputs = model(x_true_memory)
                    loss_labels = beta * criterion(true_memory_outputs, y_true_memory)
                    wandb.log({"Train/Loss_True": loss_labels})
                    loss += loss_labels
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for metric in train_metrics:
                        metric.update(y_pred=train_outputs, y_true=train_y)
            exp_time += time.perf_counter() - start_time
            for metric in train_metrics:
                metric.aggregate()

            if val_dict is not None:
                patience_counter += 1
                # evaluate the model
                if since_last_eval >= eval_every:
                    since_last_eval = 0
                    model.eval()  # Set the model to evaluation mode
                    with torch.no_grad():
                        val_outputs = model(val_exp_x)
                        for metric in val_metrics:
                            res = metric.update(y_pred=val_outputs, y_true=val_exp_y, store=False)
                            if metric.name == "Val/Loss":
                                if res < best_loss_so_far - early_stop_margin:
                                    best_loss_so_far = res
                                    patience_counter = 0
        model.eval()  # Set the model to evaluation mode
        # Add the current experience to the replay memory
        start_time = time.perf_counter()
        all_logits = model(train_exp_x).detach().cpu().numpy()
        all_logits = list(all_logits)
        all_logits_nested = np.empty(len(all_logits), dtype='object')
        all_logits_nested[:] = all_logits
        replay_memory.add_to_memory(
            train_exp_x.detach().cpu().numpy(),
            all_logits_nested,
            train_exp_y.detach().cpu().numpy())
        exp_time += time.perf_counter() - start_time
        wandb.log({"Training time": exp_time})
        classes_seen_so_far.update(new_classes)
        with torch.no_grad():
            for label in classes_seen_so_far:
                exp_test_x, exp_test_y = zip(*r_test_dict[label].values())
                exp_test_x = torch.cat([torch.from_numpy(array) for array in exp_test_x]).to(device).float()
                exp_test_y = torch.cat([torch.from_numpy(array) for array in exp_test_y]).to(device).long()
                test_outputs = model(exp_test_x)
                # we need to remap the label to the original one
                original_label = de_mapping[label]
                for metric in test_metrics:
                    if f"class_{original_label}" in metric.name:
                        metric.update(y_pred=test_outputs, y_true=exp_test_y, store=False)
                for metric in test_forgetting_metrics:
                    if f"class_{original_label}" in metric.name:
                        metric.update(y_pred=test_outputs, y_true=exp_test_y, store=False)
            del exp_test_x, exp_test_y, test_outputs
        with torch.no_grad():
            cumulative_x, cumulative_y = zip(
                *[data for label in classes_seen_so_far for data in r_test_dict[label].values()])
            cumulative_x = torch.cat([torch.from_numpy(array) for array in cumulative_x]).to(device).float()
            cumulative_y = torch.cat([torch.from_numpy(array) for array in cumulative_y]).to(device).long()
            test_outputs = model(cumulative_x)
            for metric in cumulative_test_metrics:
                metric.update(y_pred=test_outputs, y_true=cumulative_y, store=False)
            del cumulative_x, cumulative_y, test_outputs
        # SMOOTHING
        smoothing_time = 0
        with torch.no_grad():
            start_time = time.perf_counter()
            smooth_x = []
            smooth_y = []
            for run in [1, 2]:
                for label in classes_seen_so_far:
                    if run in r_test_dict[label]:
                        smooth_x.append(r_test_dict[label][run][0])
                        smooth_y.append(r_test_dict[label][run][1])
            smooth_x = torch.cat([torch.from_numpy(array) for array in smooth_x]).to(device).float()
            smooth_y = torch.cat([torch.from_numpy(array) for array in smooth_y]).to(device).long()
            non_smooth_outputs = model(smooth_x).detach().cpu().numpy()
            smooth_outputs = moving_average_by_column(non_smooth_outputs, window_size=smoothing_window)
            smoothing_time += time.perf_counter() - start_time
            smoothed_pred_accumulator.append(np.argmax(smooth_outputs, axis=1))
            pred_accumulator.append(np.argmax(non_smooth_outputs, axis=1))
            for metric in smoothened_test_metrics:
                metric.update(y_pred=torch.Tensor(smooth_outputs), y_true=smooth_y, store=False)
            del smooth_x, smooth_y
        wandb.log({"Smoothing Time": smoothing_time})

    wandb.define_metric("Observations")
    wandb.define_metric("Prediction/*", step_metric="Observations")
    wandb.define_metric("Smoothed Prediction/*", step_metric="Observations")
    # log the smoothed predictions for all experiences, one point in the dataset at a time
    max_len = max(map(len, smoothed_pred_accumulator))
    log_points_smooth = list(zip(*(l.tolist() + [None] * (max_len - len(l)) for l in smoothed_pred_accumulator)))
    log_points = list(zip(*(l.tolist() + [None] * (max_len - len(l)) for l in pred_accumulator)))
    for i in range(len(log_points)):
        wandb.log({"Observations": i})
        log_p = log_points[i]
        log_p_smooth = log_points_smooth[i]
        for exp in range(len(log_p_smooth)):
            value = log_p[exp]
            smooth_value = log_p_smooth[exp]
            if smooth_value is not None:
                wandb.log({f"Smoothed Prediction/exp_{exp}": smooth_value})
                wandb.log({f"Prediction/exp_{exp}": value})


def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    return (y_pred == y_true).float().mean()


def cross_entropy(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)


def forgetting():
    # Initialize a dictionary to hold the first value and a flag to indicate if it's the first call
    state = {'first_value': None}

    def inner_function(y_pred, y_true):
        # On the first call, store the value and set the flag to False
        value = accuracy(y_pred, y_true)
        if state['first_value'] is None:
            state['first_value'] = value
        # Compute and return the difference from the first value
        return state['first_value'] - value

    return inner_function


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_class_mapper(experiences_targets):
    # remap the classes to the order specified in the config
    if isinstance(experiences_targets[0][0], int):
        target_list = []
        for exp in experiences_targets:
            for label in exp:
                if label not in target_list:
                    target_list.append(label)
    elif isinstance(experiences_targets[0][0], tuple):
        target_list = []
        for exp in experiences_targets:
            for label, run in exp:
                if label not in target_list:
                    target_list.append(label)
    else:
        raise ValueError("Invalid experiences_targets format")
    # mapper and de-mapper
    return {label: i for i, label in enumerate(target_list)}, {i: label for i, label in enumerate(target_list)}


def map_y(y, mapping):
    return np.array([mapping[label] for label in y])


def eval_every_parser(value):
    if value == "inf":
        return np.inf
    return int(value)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Configure your experiment.')
    # Add arguments
    parser.add_argument("--data_path", type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--api_key_file', type=str, required=True, help='Path to the wandb api key json file.')
    parser.add_argument('--experiences_targets', type=ast.literal_eval,
                        help='List of lists, each one must contain the classes for that experience')
    parser.add_argument('--val_ratio', type=float, required=True, help='Validation ratio.')
    parser.add_argument('--test_ratio', type=float, required=True, help='Test ratio.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size.')
    parser.add_argument('--model', type=str, default='LSTM', help='Model type.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--window_size', type=int, required=True, help='Window size.')
    parser.add_argument('--step_size', type=int, required=True, help='Step size.')
    parser.add_argument('--memory_size', type=float, default=np.inf, help='Memory size.')
    parser.add_argument('--alpha', type=float, required=True, help='Alpha parameter for the loss.')
    parser.add_argument('--beta', type=float, required=True, help='Beta parameter for the loss.')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stop patience.')
    parser.add_argument('--early_stop_margin', type=float, default=0.001, help='Early stop margin.')
    parser.add_argument('--seed', type=int, required=True, help='Random seed.')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs.')
    parser.add_argument('--wandb_mode', type=str, default='offline', help='Wandb mode.[online, offline]',
                        choices=['online', 'offline'])
    parser.add_argument('--device', type=str, default='cpu', help='Device to use.')
    parser.add_argument('--growing_model', action='store_true', default=False,
                        help='if True, the model will grow its output layer with the number of classes.')
    parser.add_argument('--smoothing', action='store_true', default=False,
                        help='if True, results with prediction smoothing are saved')
    parser.add_argument('--run_name', type=str, help='Wandb run name.')
    parser.add_argument('--eval_every', type=eval_every_parser, default=1, help='eval every k epochs')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name.')
    parser.add_argument('--group_name', type=str, required=False, help='Name to group runs in Wandb')
    parser.add_argument('--wandb_folder', type=str, required=False, help='folder to save the data. ./wandb/ if not specified.')
    parser.add_argument('--smoothing_window', type=int, default=6, help='Window size for smoothing')
    parser.add_argument('--wandb_entity', type=str, required=True, help='Wandb entity name')

    args = parser.parse_args()
    # setup wandb
    os.environ["WANDB_MODE"] = args.wandb_mode
    with open(args.api_key_file, 'r') as file:
        j_data = json.load(file)
        api_key = j_data['api-key']

    # Step 2: Use the API key to log in to W&B
    wandb.login(key=api_key)
    config = {
        "experiences_targets": args.experiences_targets,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "batch_size": args.batch_size,
        "model": args.model,
        "learning_rate": args.learning_rate,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "memory_size": args.memory_size,
        "alpha": args.alpha,
        "beta": args.beta,
        "max_epochs": args.max_epochs,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_margin": args.early_stop_margin,
        "device": args.device,
        "growing_model": args.growing_model,
        "seed": args.seed,
        "eval_every": args.eval_every,
        "group_name": args.group_name if args.group_name is not None else "standard_group",
        "smoothing_window": args.smoothing_window
    }
    if "wandb_folder" in args and args.wandb_folder is not None:
        wandb.init(project=args.project, config=config, entity=args.wandb_entity, name=args.run_name, dir=args.wandb_folder)
    else:
        wandb.init(project=args.project, config=config, entity=args.wandb_entity, name=args.run_name)
    # define the steps otherwise the graph on W&B will follow the global step and will difficult to compare
    wandb.define_metric("Epoch")
    wandb.define_metric("Experience")

    # set seeds for reproducibility
    set_seeds(wandb.config["seed"])

    # load the data
    df = pd.read_csv(args.data_path)
    # clean useless classes
    df = df.drop(columns=features_to_drop)
    # map A,B,C,... to 0,1,2,...
    df.loc[:, class_column] = preprocessing.LabelEncoder().fit_transform(df[class_column])
    num_classes = len(df[class_column].unique())

    # normalize the data
    features_names = df.columns.tolist()
    features_names.remove(class_column)
    features_names.remove(run_column)
    num_features = len(features_names)
    # int64 features need also to be standardized, otherwise the model doesn't learn
    # First we need to change their type to float since standardizing puts them to float,
    # but you cannot assign float to an int colum
    feature_types = df[features_names].dtypes
    df = df.astype({col_name: 'float64' for col_name in features_names})
    df.loc[:, features_names] = preprocessing.StandardScaler().fit_transform(df[features_names])

    # get the windows
    train_windows, test_windows, val_windows = get_test_train_val_windowed_runs(
        df,
        "PathOrder", "Class",
        wandb.config["window_size"], wandb.config["step_size"],
        wandb.config["val_ratio"], wandb.config["test_ratio"]
    )

    # Check if CUDA is available
    if 'cuda' in wandb.config["device"]:
        if torch.cuda.is_available():
            # Check if the specified device is available
            device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
            if device_id >= torch.cuda.device_count():
                print(f"Device {args.device} is not available. Using cpu")
                wandb.config.update({
                    "device": 'cpu'
                })
        else:
            print(f"No cuda available. Using cpu")
            wandb.config.update({
                "device": 'cpu'
            })

    print(f"Using device: {wandb.config['device']}")

    # create the model
    device = wandb.config["device"]
    if wandb.config["growing_model"]:
        if isinstance(args.experiences_targets[0][0], Tuple):
            out_dim = len(set([run[0] for run in args.experiences_targets[0]]))
        else:
            out_dim = len(args.experiences_targets[0])
    else:
        out_dim = num_classes
    model = LSTM(num_features, out_dim)

    # wandb.watch(model)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    n_epochs = wandb.config["max_epochs"]

    # train the model
    derpp_train_eval_model(
        model=model,
        train_dict=train_windows,
        val_dict=val_windows,
        test_dict=test_windows,
        criterion=criterion,
        optimizer=optimizer,
        experiences_targets=args.experiences_targets,
        batch_size=wandb.config["batch_size"],
        memory_size=wandb.config["memory_size"],
        alpha=wandb.config["alpha"],
        beta=wandb.config["beta"],
        n_epochs=n_epochs,
        early_stop_patience=wandb.config["early_stop_patience"],
        early_stop_margin=wandb.config["early_stop_margin"],
        growing_model=wandb.config["growing_model"],
        smoothing_window=wandb.config["smoothing_window"],
        device=device,
        eval_every=wandb.config["eval_every"]
    )

    wandb.finish()
