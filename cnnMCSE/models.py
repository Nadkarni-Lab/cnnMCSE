import os
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class A3(nn.Module):
    def __init__(self, input_size: int=784, 
        hidden_size_one: int = 1024, 
        hidden_size_two : int = 512, 
        hidden_size_three: int = 256, 
        latent_classes: int = 2):
        super(A3, self).__init__()
        self.input_size = input_size
        self.hidden_size_one = hidden_size_one
        self.hidden_size_two = hidden_size_two
        self.hidden_size_three = hidden_size_three

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_one),
            nn.ReLU(True), 
            nn.Linear(hidden_size_one, hidden_size_two), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_two, hidden_size_three), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_three, latent_classes))

        self.decoder = nn.Sequential(
            nn.Linear(latent_classes, hidden_size_three),
            nn.ReLU(True),
            nn.Linear(hidden_size_three, hidden_size_two),
            nn.ReLU(True),
            nn.Linear(hidden_size_two, hidden_size_one),
            nn.ReLU(True),
            nn.Linear(hidden_size_one, input_size))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_2(self, x):
        x = self.encoder(x)
        return x 

class FCN(nn.Module):
    def __init__(self, input_size: int=784, 
        hidden_size_one: int = 1024, 
        hidden_size_two : int = 512, 
        hidden_size_three: int = 256, 
        output_size: int = 10):
        super(FCN, self).__init__()
        self.input_size = input_size
        self.hidden_size_one = hidden_size_one
        self.hidden_size_two = hidden_size_two
        self.hidden_size_three = hidden_size_three
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size_one),
            nn.ReLU(True), 
            nn.Linear(hidden_size_one, hidden_size_two), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_two, hidden_size_three), 
            nn.ReLU(True), 
            nn.Linear(hidden_size_three, output_size))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        return x

    def forward_2(self, x):
        x = self.encoder(x)
        return x     


class cnnAE(nn.Module):
    def __init__(self) -> None:
        super(cnnAE, self).__init__()       
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        # Decoder layers
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.t_conv1(x))
        print(x.shape)
        x = (self.t_conv2(x))
        print(x.shape)

        return x

class cnnFCN(nn.Module):
    def __init__(self) -> None:
        super(cnnFCN, self).__init__()
        # Encoder layers

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(in_features=196, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = F.relu(self.conv2(x))
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print(x.shape)
        x = (self.linear(x))
        print(x.shape)
        return x


def model_helper(model:str, initial_weights_dir:str)->nn.Module:
    """Method to return torch model. 

    Args:
        model (str): Name of the model.

    Returns:
        nn.Module: Model to return. 
    """
    if(model == "A3"):
        a3 = A3()
        initial_weights_path = os.path.join(initial_weights_dir, model + '.pt')
        torch.save(a3.state_dict(), initial_weights_path)
        return A3, initial_weights_path
    
    elif(model == "FCN"):
        fcn = FCN()
        initial_weights_path = os.path.join(initial_weights_dir, model + '.pt')
        torch.save(fcn.state_dict(), initial_weights_path)
        return FCN, initial_weights_path
    
    elif(model == "cnnAE"):
        initial_model = cnnAE()
        initial_weights_path = os.path.join(initial_weights_dir, model + '.pt')
        torch.save(initial_model.state_dict(), initial_weights_path)
        return cnnAE, initial_weights_path
    
    elif(model == "cnnFCN"):
        initial_model = cnnFCN()
        initial_weights_path = os.path.join(initial_weights_dir, model + '.pt')
        torch.save(initial_model.state_dict(), initial_weights_path)
        return cnnFCN, initial_weights_path
    
    else:
        return None
    
