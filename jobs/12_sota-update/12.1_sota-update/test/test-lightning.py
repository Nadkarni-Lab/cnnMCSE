import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from sklearn import metrics
import pandas as pd
from pytorch_lightning import LightningModule

from cnnMCSE.dataloader import dataloader_helper
from cnnMCSE.utils.helpers import generate_sample_sizes, experiment_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.zoo import transfer_helper
from cnnMCSE.metrics import get_sAUC2

class Classifier(LightningModule):
    def __init__(self, batch_size:int=4, learning_rate:float=0.01, num_class=10, zoo_model="radimagenet"):
        
        super().__init__()
        self.save_hyperparameters()
        
        # initial training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.pretrained_model = transfer_helper(zoo_model)
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True), 
            nn.Linear(1024, 512), 
            nn.ReLU(True), 
            nn.Linear(512, 256), 
            nn.ReLU(True), 
            nn.Linear(256, num_class)
        )

        self.criterion = nn.CrossEntropyLoss()
        
    
    def forward(self, x):
        # Encoder Architecture
        x = self.pretrained_model(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.encoder(x)
        x = F.log_softmax(x, dim=1)
        #x = torch.softmax(x, dim=-1)
        return x
    
    def training_step(self, batch, batch_idx):
        
        # Load and reshape the training data.
        x, y = batch

        
        # Generate predictions
        x_hat = self(x)
        
        # Generate Loss
        loss = self.criterion(x_hat, y)
        self.log('train_loss', loss)

        return loss
     
    def validation_step(self, batch, batch_idx):
        
        # Load and reshape the training data.
        x, y = batch
        
        # Generate predictions
        x_hat = self(x)
        
        if(torch.isnan(x_hat).any()):
            loss = None
            return None
        else:
            loss = self.loss_fn(x_hat, y_discrete)
            self.log('validation_loss', loss)
            validation = [validation_function(x_hat, y_discrete) for validation_function in self.validation_functions]
            self.log('val_accuracy', validation[0])
            self.log('val_TPR', validation[1])
            self.log('val_TNR', validation[2])
        
        # print(loss)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        
        # Load and reshape the training data.
        x, y = batch
        if(x.shape == [1, 128, 6]):
            x = x.transpose(1,2)
            y = y.transpose(1,2)

        
        # Generate predictions
        x_hat = self(x)
        y_discrete = (y > 0).float()
        
        if(torch.isnan(x_hat).any()):
            loss = None
            return None
        else:
            loss = self.loss_fn(x_hat, y_discrete)
            self.log('validation_loss', loss)
            validation = [validation_function(x_hat, y_discrete) for validation_function in self.validation_functions]
            self.log('val_accuracy', validation[0])
            self.log('val_TPR', validation[1])
            self.log('val_TNR', validation[2])
        
        print(loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizers = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizers
