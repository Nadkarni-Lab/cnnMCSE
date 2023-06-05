import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from sklearn import metrics
import pandas as pd

from cnnMCSE.dataloader import dataloader_helper
from cnnMCSE.utils.helpers import generate_sample_sizes, experiment_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.zoo import transfer_helper
from cnnMCSE.metrics import get_sAUC2

class Backbone(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        base_model = models.resnet50(pretrained=False, num_classes=1)
        base_model.load_state_dict(torch.load("/sc/arion/projects/EHR_ML/gulamf01/cnnMCSE/jobs/12_sota-update/12.1_sota-update/weights/RadImageNet-ResNet50_notop_torch.pth"))
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])

        if(freeze):
            for p in self.backbone.parameters():
                p.requires_grad = False
                        
    def forward(self, x):
        return self.backbone(x)
    
class Classifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        #self.linear = nn.Linear(2048, num_class)
        self.encoder = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(True), 
        nn.Linear(1024, 512), 
        nn.ReLU(True), 
        nn.Linear(512, 256), 
        nn.ReLU(True), 
        nn.Linear(256, num_class))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.encoder(x)
        x = F.log_softmax(x, dim=1)
        #x = torch.softmax(x, dim=-1)
        return x

    def predict(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.encoder(x)
        x = F.softmax(x, dim=1)
        return x

class Classifier2(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.drop_out = nn.Dropout()
        self.linear = nn.Linear(2048, num_class)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.linear(x)
        #x = torch.softmax(x, dim=-1)
        return x

generator = torch.Generator().manual_seed(42)

# Get Initial Weights Directory. 
initial_weights_dir = '/sc/arion/projects/EHR_ML/gulamf01/cnnMCSE/jobs/12_sota-update/12.1_sota-update/weights'
estimand = "tlRADFCN"
estimand, initial_estimand_weights_path = model_helper(model=estimand, initial_weights_dir=initial_weights_dir)

# Dataset List. 
dataset_list = ['SOTA-ethnicity-pneumothorax']
experiment = "MIMIC"
start_seed = 42
root_dir = '/sc/arion/projects/EHR_ML/gulamf01/cnnMCSE/data'
using_pretrained = True
batch_size = 64
num_workers = 1
zoo_model = "radimagenet"
n_epochs = 30
num_classes = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

for current_dataset in dataset_list:
    dataset_dict = experiment_helper(experiment=experiment, dataset=current_dataset, root_dir=root_dir, tl_transforms=using_pretrained, start_seed=start_seed)
    
    train_test_match = dataset_dict['train_test_match']

    sample_sizes = generate_sample_sizes(
        max_sample_size=40000, 
        log_scale=2, 
        min_sample_size=30000, 
        absolute_scale=False
    )
    print(sample_sizes)

    # Classifier. 
    current_model = Classifier(num_class=2)
    current_model.load_state_dict(torch.load(initial_estimand_weights_path))

    # Parallelize current model. 
    #print("Parallelize current model.")
    current_model = nn.DataParallel(current_model)
    current_model.to(device)

    pretrained_model = Backbone(freeze=True)
    pretrained_model = pretrained_model.to(device=device) 

    criterion = nn.CrossEntropyLoss()
    optimizer_model = optim.AdamW(current_model.parameters(), lr=0.001)
    print(dataset_dict)
    for trainset, testset in train_test_match:
        training_data = dataset_dict[trainset]
        test_data = dataset_dict[testset]
        print(test_data)
        print(testset)
        for sample_size in sample_sizes:
            train_subset, validation_subset = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
            trainloader = torch.utils.data.DataLoader(train_subset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=True)
            
            valloader = torch.utils.data.DataLoader(validation_subset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=True)
            testloader = torch.utils.data.DataLoader(test_data,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    pin_memory=True)
            current_model.train()
            for epoch in range(n_epochs):
                
                
                for j, data in enumerate(trainloader):
                    # Get data
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    if(pretrained_model):
                        inputs = pretrained_model(inputs)


                    # Zero parameter gradients
                    optimizer_model.zero_grad()

                    # Forward + backward + optimize
                    outputs = current_model(inputs)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_model.step()
            
                print("Get validation AUC")
                current_model.eval()
                roc_df, preds_df = get_sAUC2(
                    model=current_model,
                    loader=valloader,
                    zoo_model=zoo_model
                )
                print('Validation AUC', roc_df)
           
            testloader = torch.utils.data.DataLoader(testset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True)
            
            roc_df, preds_df = get_sAUC2(
                    model=current_model,
                    loader=testloader,
                    zoo_model=zoo_model
            )
            print('Test AUC', roc_df)
