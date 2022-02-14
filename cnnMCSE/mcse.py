import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.optim as optim
import numpy as np

from sklearn.datasets import make_classification
from scipy.interpolate import UnivariateSpline
import numpy as np


BATCH_SIZE = 4
EPOCH = 1
NUM_WORKERS = 2

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

def get_derivative(A3_loss_raw, sample_sizes):
    log_sample_sizes = np.log(sample_sizes)
    A3_loss_mean = [np.mean(a3_loss) for a3_loss in A3_loss_raw]
    y_spl = UnivariateSpline(log_sample_sizes, A3_loss_mean,s=0,k=2)
    y_spl_1d = y_spl.derivative(n=1)
    y_spl_2d = y_spl.derivative(n=2)
    
    return y_spl, y_spl_1d, y_spl_2d

def get_inflection_point(spl, spl_2D, sample_sizes):
    log_sample_sizes = np.log(sample_sizes)
    sample_size_peak = np.argmax(spl_2D(log_sample_sizes))
    peak_A3 = spl(log_sample_sizes[sample_size_peak])
    error_down = spl(log_sample_sizes[sample_size_peak+1])
    error_up   = spl(log_sample_sizes[sample_size_peak-1])
    return peak_A3, error_down, error_up

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

def msse_predict(
    max_sample_size : int = 1000, 
    log_scale: int = 2, 
    min_sample_size: int = 64, 
    absolute_scale = None,
    n_informative = 78, 
    n_features = 784, 
    n_classes=10,
    bootleg=2,
    hidden_size_one: int = 1024, 
    hidden_size_two : int = 512, 
    hidden_size_three: int = 256, 
    latent_classes: int = 2):
    
    # generate sample sizes. 
    sample_sizes = generate_sample_sizes(max_sample_size=max_sample_size, log_scale=log_scale, min_sample_size=min_sample_size, absolute_scale=absolute_scale)
    
    # generate dataset. 
    sample_dataset = make_classification(n_samples=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes)
    tensor_x = torch.Tensor(sample_dataset[0]) 
    tensor_y = torch.LongTensor(sample_dataset[1]) 
    sample_dataset = TensorDataset(tensor_x,tensor_y) 
    
    # collect bootstraps. 
    A3_losses = list()
    for sample_size in sample_sizes:
        model_current_sample_size, training_loss_current_sample_size = get_models_bootleg_A3(
                            sample_size=sample_size,
                            trainset=sample_dataset,
                            bootleg=bootleg,
                            input_size=n_features,
                            hidden_size_one = hidden_size_one, 
                            hidden_size_two = hidden_size_two, 
                            hidden_size_three = hidden_size_three, 
                            latent_classes = latent_classes)
        A3_losses.append(training_loss_current_sample_size)


    print(A3_losses)
    print(sample_sizes)
    A3_loss_final = list()
    sample_size_final = list()

    for sample_size, A3_loss in zip(sample_sizes, A3_losses):
        A3_loss_2 = [x for x in A3_loss if np.isnan(x) == False]
        if(len(A3_loss_2) != 0):
            A3_loss_final.append(A3_loss_2)
            sample_size_final.append(sample_size)

    print(A3_loss_final)
    print(sample_size_final)

    y_spl, y_spl_1D, y_spl_2D         = get_derivative(A3_loss_raw=A3_loss_final, sample_sizes=sample_size_final)
    y_infl, y_infl_erd, y_infl_eru = get_inflection_point(y_spl, y_spl_2D, sample_sizes=sample_size_final)

    msse_estimate = np.exp(y_infl)
    print(msse_estimate)
    return msse_estimate
    # pass




if __name__ == '__main__':
    msse_predict()
    # 1. Initialize sample size bootstrap - done
    # 2. Change autoenocder from model passed in parameters to hyperparameters. - done
    # 3. 
    # msse_predict()