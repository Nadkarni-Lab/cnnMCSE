"""Package to calculate sample size estimates for convolutional neural networks. 
"""
import os
import gc
from random import sample
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split

from sklearn.datasets import make_classification
from scipy.interpolate import UnivariateSpline

from cnnMCSE.models import FCN, A3
from cnnMCSE.metrics import metric_helper
from cnnMCSE.utils.zoo import transfer_helper


BATCH_SIZE = 4
EPOCH = 1
NUM_WORKERS = 2



def get_estimators(
    model,
    training_data,
    sample_size:int,
    initial_weights:str, 
    batch_size:int=1,
    bootstraps:int=1,
    start_seed:int=42,
    shuffle:bool=False,
    num_workers:int=1,
    zoo_model:str=None,
    frequency:bool=False,
    stratified:bool=False):
    """Method to get estimators for convergence samples. 

    Args:
        model (_type_): A model. 
        training_data (_type_): Training data. 
        sample_size (int): Sample size to estimate at. 
        batch_size (int, optional): Batch size. Defaults to 4.
        bootstraps (int, optional): Number of bootstraps. Defaults to 1.
        start_seed (int, optional): Seed. Defaults to 42.
        shuffle (bool, optional): Shuffle the dataset. Defaults to False.
        num_workers (int, optional): Number of workers. Defaults to 1.
        frequency (bool, optional): Whether to calculate frequencies or not. 

    Returns:
        list: List of losses. 
    """

    # Determine which device is being used. 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Generate zoo models. 
    if(zoo_model):
        pretrained_model = transfer_helper(zoo_model)
        pretrained_model = pretrained_model.to(device=device)   
    else:
        pretrained_model = None         
    # run across all the bootstraps
    losses = list()
    train_subsets = list()
    s_losses = {}

    for i in range(bootstraps):
        print("Running loop ", i)

        # Create a generator for replicability. 
        print("Generating generator")
        generator = torch.Generator().manual_seed(start_seed+i)

        # generate a unique training subset.
        print("Creating training subset")
        train_subset, _ = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
        train_subsets.append(train_subset)
        

        # Create a training dataloader. 
        print("Create a training dataloader. ")
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True)
        # Initialize current model. 
        print(f"Initialize current model. {initial_weights}")
        if(zoo_model): 
            current_model = model(input_size=1000)
        else:
            current_model = model()
        current_model.load_state_dict(torch.load(initial_weights))

        # Parallelize current model. 
        print("Parallelize current model. ")
        current_model = nn.DataParallel(current_model)
        current_model.to(device)

        # Set model in training mode. 
        print("Set model in training mode. ")
        current_model.train()

        # Generate mean-squared error loss criterion
        print("Generate mean-squared error loss criterion.")
        if(stratified):
            s_criterion = nn.MSELoss(reduction='none')
            
        criterion = nn.MSELoss()
        #else:
        #   

        # Optimize model with stochastic gradient descent. 
        print("Optimize model with stochastic gradient descent. ")
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.01, momentum=0.9)

        # Set up training loop
        running_loss = 0.0
        s_running_loss = dict()
        
        # iterate over training subset. 
        print("Running dataset ")
        print(trainloader)
        print(train_subset)
        for j, data in enumerate(trainloader):
            print("Testing data ", j)
            print('Data', data)
            #print("Running batch" , i)

            # Get data
            inputs, labels = data
            #inputs = inputs.flatten()
            inputs, labels = inputs.to(device), labels.to(device)

            if(pretrained_model):
                inputs = pretrained_model(inputs)

            # Zero parameter gradients
            optimizer_model.zero_grad()

            # Forward + backward + optimize
            print(inputs.shape)
            outputs = current_model(inputs)
            print('Output shape', outputs.shape)
            # Accomodate for intra-model flattening. 
            inputs = inputs.reshape(outputs.shape)
            print('Input shape', inputs.shape)
            loss = criterion(outputs, inputs)

            if(stratified):
                print("Running stratified loss...")
                s_loss = s_criterion(outputs, inputs)
                labels = labels.tolist()
                s_loss_dict = metric_helper(metric_type="sloss", s_loss=s_loss, labels=labels, models=None)

            loss.backward()
            optimizer_model.step()

            running_loss += loss.item()

            print("Adding losses to running losses...")
            for key, value in s_loss_dict.items():
                if(key in s_running_loss):
                    s_running_loss[key] += s_loss_dict[key][0]
                else:
                    s_running_loss[key] = s_loss_dict[key][0]


        # Add loss
        loss = running_loss / sample_size
        losses.append(float(loss))

        if(stratified):
            for key, value in s_running_loss.items():
                s_loss = value / sample_size
                if(key in s_losses):
                    s_losses[key].append(s_loss)
                else:
                    s_losses[key] = [s_loss]


    if(frequency == True):
        loss_dict = {
            'estimators': losses
        }
        loss_df = pd.DataFrame(loss_dict)
        frequency_df = metric_helper(models=None, metric_type="frequencies", datasets=train_subsets, num_workers=0)

        merged_df = pd.concat([loss_df, frequency_df], axis=1)
        
        if(stratified):
            s_mcse_df = pd.DataFrame.from_dict(s_losses, orient='index', columns = ['s_estimators'])
            s_mcse_df = s_mcse_df.reset_index()
            s_mcse_df['label']  = s_mcse_df['index']
            print(s_mcse_df)
            print(merged_df)
            merged_df = merged_df.merge(s_mcse_df, on='label')

        merged_df['sample_size'] = sample_size
        return merged_df
    
    else:
        return losses

    gc.collect()
    return merged_df


def get_estimands(
    model,
    training_data,
    validation_data,
    sample_size,
    initial_weights:str,
    batch_size:int=1,
    bootstraps:int=1,
    start_seed:int=42,
    shuffle:bool=False,
    metric_type:str="AUC",
    num_workers:int=1,
    zoo_model:str=None
    ):
    """Method to generate estimands. 

    Args:
        model (_type_): nn.Module. 
        training_data (_type_): Training data. 
        validation_data (_type_): Validation data. 
        sample_size (_type_): Sample size. 
        initial_weights (str): Initial weights. 
        batch_size (int, optional): Batch size. Defaults to 4.
        bootstraps (int, optional): Number of bootstraps. Defaults to 1.
        start_seed (int, optional): Start seed. Defaults to 42.
        shuffle (bool, optional): Shuffle. Defaults to False.
        metric_type (str, optional): Metric type. Defaults to "AUC".

    Returns:
        list: List of estimation. 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if(zoo_model):
        pretrained_model = transfer_helper(zoo_model)
        pretrained_model = pretrained_model.to(device=device) 
    else:
        pretrained_model = None

    print("Getting estimands")
    models = list()
    model_paths = list()
    metrics = list()
    for i in range(bootstraps):
        print("Running estimands ", i)
        # Create a generator for replicability. 
        print("Create a generator for replicability.")
        generator = torch.Generator().manual_seed(start_seed+i)

        # generate a unique training subset.
        print("generate a unique training subset..")
        train_subset, _ = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
        
        # Create a training dataloader. 
        print("Create a training dataloader")
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True)
        # Initialize current model. 
        print("Initialize current model. ")
        if(zoo_model): 
            current_model = model(input_size=1000)
        else:
            current_model = model()
        current_model.load_state_dict(torch.load(initial_weights))

        # Parallelize current model. 
        print("Parallelize current model.")
        current_model = nn.DataParallel(current_model)
        current_model.to(device)

        # Set model in training mode. 
        print("Set model in training mode")
        current_model.train()

        # Assign CrossEntropyLoss and stochastic gradient descent optimizer. 
        print("Run cross entropy loss")
        criterion = nn.CrossEntropyLoss()
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.01, momentum=0.9)

        # Train the model.  
        print("Running loop for estimands")
        for j, data in enumerate(trainloader):
            print("Testing data ", j)
            # Get data
            inputs, labels = data
            #inputs = torch.flatten(inputs, start_dim=1)
            inputs, labels = inputs.to(device), labels.to(device)

            if(pretrained_model):
                inputs = pretrained_model(inputs)


            # Zero parameter gradients
            optimizer_model.zero_grad()

            # Forward + backward + optimize
            outputs = current_model(inputs)

            print(outputs.shape)
            print(labels.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_model.step()
        
        models.append(current_model)

    print("Evaluating models... ")
    print(len(models))
    metrics = metric_helper(
        models = models,
        dataset=validation_data,
        metric_type=metric_type,
        num_workers=num_workers,
        zoo_model=zoo_model
    )
    gc.collect()
    return metrics