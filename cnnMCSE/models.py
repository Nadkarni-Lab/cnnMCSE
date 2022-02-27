import os
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as mo

class A3(nn.Module):
    """Fully connected network autoencoder. 
    """
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
    """Fully connected neural network classifier. 
    """
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
    """Convolutional neural network autoencoder. 
    """
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
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = (self.t_conv2(x))
        return x

class cnnFCN(nn.Module):
    """Convolutional neural network classifier. 
    """
    def __init__(self) -> None:
        super(cnnFCN, self).__init__()
        # Encoder layers

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(in_features=196, out_features=10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = (self.linear(x))
        return x

class AlexNetFCN(nn.Module):
    def __init__(self, input_dim:list = [3, 224,224], num_classes: int = 10) -> None:
        super(AlexNetFCN, self).__init__()
        self.input_dim = input_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetAE(nn.Module):
    def __init__(self, input_dim: list = [3, 224, 224], latent_space: int = 10) -> None:
        super(AlexNetAE, self).__init__()
        self.input_dim = input_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_dim[0], 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, latent_space),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3*224*224)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.decoder(x)
        x = x.reshape(self.input_dim)
        return x

def alexnetFCN(num_classes:int=10):
    """Transfer learning classifier. 

    Args:
        num_classes (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    return mo.alexnet(num_classes=num_classes)

class alexnetAE(nn.Module):
    def __init__(self) -> None:
        super(alexnetAE, self).__init__()
        





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
    
