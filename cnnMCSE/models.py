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
    def __init__(self, input_size: int=784, 
        hidden_size_one: int = 1024, 
        hidden_size_two : int = 512, 
        hidden_size_three: int = 256, 
        output_size: int = 10):
        super(A3, self).__init__()
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
    


def model_helper(model:str)->nn.Module:
    """Method to return torch model. 

    Args:
        model (str): Name of the model.

    Returns:
        nn.Module: Model to return. 
    """
    if(model == "A3"):
        return A3
    
    elif(model == "FCN"):
        return FCN
    
    else:
        return None
    
