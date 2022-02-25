"""Package to calculate sample size estimates for convolutional neural networks. 
"""
import os
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split

from sklearn.datasets import make_classification
from scipy.interpolate import UnivariateSpline

from cnnMCSE.models import FCN, A3
from cnnMCSE.metrics import metric_helper


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
    num_workers:int=1):
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

    Returns:
        list: List of losses. 
    """


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # run across all the bootstraps
    losses = list()
    for i in range(bootstraps):
        print("Running loop ", i)

        # Create a generator for replicability. 
        print("Generating generator")
        generator = torch.Generator().manual_seed(start_seed+i)

        # generate a unique training subset.
        print("Creating training subset")
        train_subset, _ = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
        
        # Create a training dataloader. 
        print("Create a training dataloader. ")
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True)
        # Initialize current model. 
        print("Initialize current model. ")
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
        criterion = nn.MSELoss()

        # Optimize model with stochastic gradient descent. 
        print("Optimize model with stochastic gradient descent. ")
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.01, momentum=0.9)

        # Set up training loop
        running_loss = 0.0
        
        # iterate over training subset. 
        print("Running dataset ")
        for j, data in enumerate(trainloader):
            print("Testing data ", j)
            #print("Running batch" , i)

            # Get data
            inputs, labels = data
            #inputs = inputs.flatten()
            inputs, labels = inputs.to(device), labels.to(device)


            # Zero parameter gradients
            optimizer_model.zero_grad()

            # Forward + backward + optimize
            outputs = current_model(inputs)

            # Accomodate for intra-model flattening. 
            inputs = inputs.reshape(outputs.shape)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer_model.step()

            running_loss += loss.item()

        # Add loss
        loss = running_loss / sample_size
        losses.append(float(loss))

    gc.collect()
    return losses


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
    num_workers:int=1
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

            # Zero parameter gradients
            optimizer_model.zero_grad()

            # Forward + backward + optimize
            outputs = current_model(inputs)
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
        num_workers=num_workers
    )
    gc.collect()
    return metrics