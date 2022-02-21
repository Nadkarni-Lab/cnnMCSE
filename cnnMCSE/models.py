import os
import torch
import torch.nn as nn

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_2(self, x):
        x = self.encoder(x)
        return x 


class FCN(nn.Module):
    """
    A simple fully connected network used to approximate the quantify the
    learnability of datasets at different sample sizes.
    """
    def __init__(self):
        """
        Initialization of the fully connected network with an input dimension of
        784, a linear layer with 36 hidden units, ReLU activation, followed by
        a linear layer with 16 hidden units, ReLU activation, followed by
        the output layer.
        """
        super(FCN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 36),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(36, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        """
        Forward pass of the fully connected network.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),  -1)
        out = self.fc(out)
        return out

# class FCN2(nn.Module):
#     """
#     A simple fully connected network used to approximate the quantify the
#     learnability of datasets at different sample sizes.
#     """
#     def __init__(self):
#         """
#         Initialization of the fully connected network with an input dimension of
#         784, a linear layer with 36 hidden units, ReLU activation, followed by
#         a linear layer with 16 hidden units, ReLU activation, followed by
#         the output layer.
#         """
#         super(FCN, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Linear(784, 36),
#             nn.ReLU(),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(36, 16),
#             nn.ReLU()
#         )
#         self.fc = nn.Linear(16, 10)

#     def forward(self, x):
#         """
#         Forward pass of the fully connected network.
#         """
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0),  -1)
#         out = self.fc(out)
#         return out

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
        x = self.encoder(x)
        return x

    def forward_2(self, x):
        x = self.encoder(x)
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
    
    else:
        return None
    
