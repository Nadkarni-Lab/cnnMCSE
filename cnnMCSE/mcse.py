"""Package to calculate sample size estimates for convolutional neural networks. 
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from sklearn.datasets import make_classification
from scipy.interpolate import UnivariateSpline

from cnnMCSE.models import FCN, A3


BATCH_SIZE = 4
EPOCH = 1
NUM_WORKERS = 2


def get_model_sample_size_A3(sample_size, trainset,
        input_size, 
        hidden_size_one, 
        hidden_size_two, 
        hidden_size_three, 
        latent_classes,
        bootleg=1):
    # initial_weights = torch.save(model.state_dict(), MODEL_PATH)

    for i in range(bootleg):
        # print('Training model', (i+1))

        # Get the training subset
        indices = np.random.randint(len(trainset), size=sample_size)
        # print(indices)
        train_subset = torch.utils.data.Subset(trainset, indices)

        # Set up the training loader
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)#,
                                              #num_workers=NUM_WORKERS)
        # Initialize training weights
        current_model = A3(
            input_size=input_size, 
            hidden_size_one=hidden_size_one, 
            hidden_size_two=hidden_size_two, 
            hidden_size_three=hidden_size_three,
            latent_classes=latent_classes
        )
        #current_model.load_state_dict(torch.load(MODEL_PATH))

        current_model = nn.DataParallel(current_model)
        #current_model.to(DEVICE)


        current_model.train()
        criterion = nn.MSELoss()
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.01, momentum=0.9)

        # Set up training loop
        running_loss = 0.0

        for data in (train_subset):
            # Get data
            inputs, labels = data
            inputs = inputs.flatten()

            # add noise
            #noise = torch.randn_like(inputs)
            #inputs += noise



            # inputs = inputs.to(DEVICE)

            # Zero parameter gradients
            optimizer_model.zero_grad()

            # Forward + backward + optimize
            outputs = current_model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer_model.step()


            running_loss += loss.item()

        # Add model to models and loss to training losses

        loss = running_loss / sample_size
        # print(loss)
        # print(current_model.state_dict())

    return current_model.state_dict(), loss


def get_models_bootleg_A3(sample_size, trainset, bootleg, 
    input_size,
    hidden_size_one, 
    hidden_size_two, 
    hidden_size_three, 
    latent_classes):
    models = []
    models_train_loss = []

    for i in range(bootleg):
        # print('Training sample size', sample_size, 'version: ', i)
        current_model, train_loss = get_model_sample_size_A3(
            sample_size=sample_size,
            trainset=trainset,
            input_size=input_size,
            hidden_size_one = hidden_size_one, 
            hidden_size_two = hidden_size_two, 
            hidden_size_three = hidden_size_three, 
            latent_classes = latent_classes)
        models.append(current_model)
        models_train_loss.append(train_loss)

    return models, models_train_loss

def get_model_sample_size(sample_size, 
    trainset, 
    input_size,
    hidden_size_one, 
    hidden_size_two, 
    hidden_size_three, 
    output_size,
    noise_factor=0):


    # initial_weights = torch.save(model.state_dict(), MODEL_PATH)


    for i in range(bootleg):
        # print('Training model', (i+1))

        # Get the training subset
        indices = np.random.randint(len(trainset), size=sample_size)
        # print(indices)
        train_subset = torch.utils.data.Subset(trainset, indices)

        # Set up the training loader
        trainloader = torch.utils.data.DataLoader(train_subset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=NUM_WORKERS)
        
        
        
        # Initialize training weights
        current_model = FCN(
            input_size = input_size,
            hidden_size_one = hidden_size_one,
            hidden_size_two = hidden_size_two,
            hidden_size_three = hidden_size_three,
            output_size=output_size
        )
        #current_model.load_state_dict(torch.load(MODEL_PATH))

        #current_model = nn.DataParallel(model)
        #current_model.to(DEVICE)


        current_model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer_model = optim.SGD(current_model.parameters(), lr=0.01, momentum=0.9)

        # Set up training loop
        running_loss = 0.0

        for i, data in enumerate(trainloader):
            # Get data
            inputs, labels = data
            inputs = torch.flatten(inputs, start_dim=1)


            #inputs, labels = inputs.to(DEVICE), (labels).to(DEVICE)

            # Zero parameter gradients
            optimizer_model.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_model.step()


            running_loss += loss.item()

        # Add model to models and loss to training losses

        loss = running_loss / sample_size
        # print(loss)
        # print(current_model.state_dict())

    return current_model.state_dict(), loss

def get_models_bootleg(model_name, sample_size, trainset, bootleg,noise_factor=1):
    models = []
    models_train_loss = []

    for i in range(bootleg):
        print('Training sample size', sample_size, 'version: ', i)
        current_model, train_loss = get_model_sample_size(model=FCN(),
                                              sample_size=sample_size,
                                              trainset=trainset,
                                              noise_factor=noise_factor)
        #print(current_model)
        models.append(current_model)
        models_train_loss.append(train_loss)

    return models, models_train_loss



def generate_sample_sizes(max_sample_size : int = 5000, log_scale: int = 2, min_sample_size: int = 64, absolute_scale = None):
    sample_size_list = list()

    if(absolute_scale == None):
        sample_size = int(max_sample_size)
        while sample_size > min_sample_size:
            sample_size_list.append(sample_size)
            sample_size = int(sample_size / log_scale)
        sample_size_list.append(min_sample_size)
    
    else:
        for sample_size in range(min_sample_size, max_sample_size, absolute_scale):
            sample_size_list.append(sample_size)
        sample_size_list.append(max_sample_size)
    sample_size_list.sort()
    print(sample_size_list)
    return sample_size_list



if __name__ == '__main__':
    msse_predict()
    # 1. Initialize sample size bootstrap - done
    # 2. Change autoenocder from model passed in parameters to hyperparameters. - done
    # 3. 
    # msse_predict()