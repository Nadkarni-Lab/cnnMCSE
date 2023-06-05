import os
import argparse
import yaml
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from sklearn import metrics
import pandas as pd
import torchmetrics


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger



from cnnMCSE.dataloader import dataloader_helper, weighted_sampler
from cnnMCSE.utils.helpers import generate_sample_sizes, experiment_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.zoo import transfer_helper
from cnnMCSE.metrics import get_sAUC2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

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

class Classifier(LightningModule):
    def __init__(self, batch_size:int=4, learning_rate:float=0.01, num_class=2, zoo_model="radimagenet"):
        
        super().__init__()
        self.save_hyperparameters()
        
        # initial training parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.pretrained_model = transfer_helper(zoo_model)
        self.drop_out = nn.Dropout(p=0.5)
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
        self.accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=num_class)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_class)
        self.multiclass_auroc_metric = torchmetrics.classification.MulticlassAUROC(num_classes=num_class)
        self.auprc_metric = torchmetrics.AveragePrecision(task="multiclass", num_classes=num_class)
        self.num_classes = num_class
        

        # Initialize Validation and Test Dictionaries. 
        self.validation_step_outputs = list()
        self.validation_step_dict = {
            'val_auroc': [],
            'val_tp'  : [],
            'val_tn'  : [],
            'val_fp'  : [],
            'val_fn'  : [],
            'val_loss' : [],
            'val_accuracy': [],
            'val_auprc' : []
        }

        self.test_step_dict = {
            'test_auroc': [],
            'test_tp'  : [],
            'test_tn'  : [],
            'test_fp'  : [],
            'test_fn'  : [],
            'test_loss' : [],
            'test_accuracy': [],
            'test_auprc': []
        }
    
    def forward(self, x):
        # Encoder Architecture
        x = self.pretrained_model(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.encoder(x)
        x = F.log_softmax(x, dim=1)
        #x = torch.softmax(x, dim=-1)
        return x
    
    def predict(self, x):
        # Prediction. 
        x = self.pretrained_model(x)
        x = x.view(x.size(0), -1)
        x = self.drop_out(x)
        x = self.encoder(x)
        x = F.softmax(x, dim=1)
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
        x_hat   = self(x)
        x_preds = self.predict(x)


        # Calculate Metrics. 
        val_loss = self.criterion(x_hat, y)
        val_auroc = self.auroc_metric(x_preds, y)
        val_accuracy = self.accuracy_metric(x_preds, y)
        val_auprc = self.auprc_metric(x_preds, y)
        
        # Get TPR and TNR rates. 
        confusion_matrix = self.confmat(x_preds, y)

        # Calculate TPR, TNR, FPR, and FNR for each class
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]

      

        self.validation_step_dict['val_auroc'].append(val_auroc)
        self.validation_step_dict['val_auprc'].append(val_auprc)
        self.validation_step_dict['val_loss'].append(val_loss)
        self.validation_step_dict['val_accuracy'].append(val_accuracy)
        self.validation_step_dict['val_tp'].append(tp)
        self.validation_step_dict['val_tn'].append(tn)
        self.validation_step_dict['val_fp'].append(fp)
        self.validation_step_dict['val_fn'].append(fn)
        self.validation_step_outputs.append(val_auroc)

    
    def on_validation_epoch_end(self):
        # outputs = self.validation_step_outputs
        val_auc = torch.stack(self.validation_step_dict['val_auroc']).mean() 
        val_auprc = torch.stack(self.validation_step_dict['val_auprc']).mean() 
        val_loss = torch.stack(self.validation_step_dict['val_loss']).mean() 
        val_accuracy = torch.stack(self.validation_step_dict['val_accuracy']).mean() 
        tp = torch.stack(self.validation_step_dict['val_tp']).sum() 
        tn = torch.stack(self.validation_step_dict['val_tn']).sum() 
        fp = torch.stack(self.validation_step_dict['val_fp']).sum() 
        fn = torch.stack(self.validation_step_dict['val_fn']).sum() 

        val_tpr = tp / (tp + fn)
        val_tnr = tn / (tn + fp)
        val_fpr = fp / (tn + fp)
        val_fnr = fn / (tp + fn)

        # Log validation. 
        self.log('epoch_val_auroc', val_auc, sync_dist=True)
        self.log('epoch_val_auprc', val_auprc, sync_dist=True)
        self.log('epoch_val_loss', val_loss, sync_dist=True)        
        self.log("epoch_val_accuracy", val_accuracy, sync_dist=True)
        self.log("epoch_val_TPR", val_tpr, sync_dist=True)
        self.log("epoch_val_TNR", val_tnr, sync_dist=True)
        self.log("epoch_val_FPR", val_fpr, sync_dist=True)
        self.log("epoch_val_FNR", val_fnr, sync_dist=True)

        # Clear Items. 
        self.validation_step_outputs = list()
        for key in self.validation_step_dict:
            self.validation_step_dict[key] = list()

        return {'epoch_val_auroc': val_auc}
 
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):
        
        # Load and reshape the training data.
        x, y = batch
        
        # Generate predictions
        x_hat   = self(x)
        x_preds = self.predict(x)

        # Log validation loss.
        loss = self.criterion(x_hat, y)


        # Calculate Metrics. 
        test_loss = self.criterion(x_hat, y)
        test_auroc = self.auroc_metric(x_preds, y)
        test_auprc = self.auprc_metric(x_preds, y)
        test_accuracy = self.accuracy_metric(x_preds, y)
        
        # Get TPR and TNR rates. 
        confusion_matrix = self.confmat(x_preds, y)

        # Calculate TPR, TNR, FPR, and FNR for each class
        tp = confusion_matrix[1][1]
        tn = confusion_matrix[0][0]
        fp = confusion_matrix[0][1]
        fn = confusion_matrix[1][0]

       
        self.test_step_dict['test_auroc'].append(test_auroc)
        self.test_step_dict['test_auprc'].append(test_auroc)
        self.test_step_dict['test_loss'].append(test_loss)
        self.test_step_dict['test_accuracy'].append(test_accuracy)
        self.test_step_dict['test_tp'].append(tp)
        self.test_step_dict['test_tn'].append(tn)
        self.test_step_dict['test_fp'].append(fp)
        self.test_step_dict['test_fn'].append(fn)
        
        return self.test_step_dict

    def on_test_epoch_end(self):
        # outputs = self.validation_step_outputs
        output_dict = {}
        output_dict['test_auc'] = torch.stack(self.test_step_dict['test_auroc']).mean()
        output_dict['test_auprc'] = torch.stack(self.test_step_dict['test_auprc']).mean()  
        output_dict['test_loss'] = torch.stack(self.test_step_dict['test_loss']).mean() 
        output_dict['test_accuracy'] = torch.stack(self.test_step_dict['test_accuracy']).mean() 
        
        # Calculate TP, TN, FP, FN (R)
        tp = torch.stack(self.test_step_dict['test_tp']).sum() 
        tn = torch.stack(self.test_step_dict['test_tn']).sum() 
        fp = torch.stack(self.test_step_dict['test_fp']).sum() 
        fn = torch.stack(self.test_step_dict['test_fn']).sum() 
        test_tpr = tp / (tp + fn)
        test_tnr = tn / (tn + fp)
        test_fpr = fp / (tn + fp)
        test_fnr = fn / (tp + fn)

        output_dict['test_tpr'] = test_tpr 
        output_dict['test_tnr'] = test_tnr
        output_dict['test_fpr'] = test_fpr 
        output_dict['test_fnr'] = test_fnr 
        

        # Log test. 
        self.log('epoch_test_auroc', output_dict['test_auc'], sync_dist=True)
        self.log('epoch_test_auroc', output_dict['test_auprc'], sync_dist=True)
        self.log('epoch_test_loss', output_dict['test_loss'], sync_dist=True)        
        self.log("epoch_test_accuracy", output_dict['test_accuracy'], sync_dist=True)
        self.log("epoch_test_TPR", output_dict['test_tpr'], sync_dist=True)
        self.log("epoch_test_TNR", output_dict['test_tnr'], sync_dist=True)
        self.log("epoch_test_FPR", output_dict['test_fpr'], sync_dist=True)
        self.log("epoch_test_FNR", output_dict['test_fnr'] , sync_dist=True)

        # Clear Items. 
        return output_dict
    
    def configure_optimizers(self):
        optimizers = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizers


def main(initial_weights_dir,
         estimand,
         dataset_list,
         experiment,
         root_dir,
         using_pretrained,
         batch_size,
         num_workers,
         zoo_model,
         max_epochs,
         num_classes,
         checkpoint_path,
         save_dir,
         auto_lr_find,
         num_gpus,
         num_nodes,
         refresh_rate,
         accelerator,
         auto_scale_batch_size,
         checkpoint_interval,
         gradient_clip_val,
         seed,
         output_directory,
         max_sample_size,
         min_sample_size,
         log_scale,
         config:str=None
         ):
    estimand, initial_estimand_weights_path = model_helper(model=estimand, initial_weights_dir=initial_weights_dir)

    # Generate seed. 
    print("Using seed. ", seed)
    seed = int(seed)
    for current_dataset in dataset_list:
        dataset_dict = experiment_helper(experiment=experiment, dataset=current_dataset, root_dir=root_dir, tl_transforms=using_pretrained, start_seed=seed)
        
        train_test_match = dataset_dict['train_test_match']

        # Isolate training datasets and testing datasets
        training_set = list() 
        testing_set = list()
        for trainset, testset in train_test_match:
            #training_data = dataset_dict[trainset]
            #test_data = dataset_dict[testset]
            training_set.append(trainset)
            testing_set.append(testset)
        
        # Get unique items. 
        training_set = list(set(training_set))
        testing_set = list(set(testing_set))

        print("Training set", training_set)
        print("Testing set", testing_set)
        
        # Get test dataloaders
        test_dataloaders = list()
        
        for item in testing_set:
            test_data = dataset_dict[item]
            print('Adding item', item)
            testloader = torch.utils.data.DataLoader(test_data,
                                                            batch_size=batch_size,
                                                            num_workers=num_workers,
                                                            pin_memory=True)
            test_dataloaders.append(testloader)



        sample_sizes = generate_sample_sizes(
            max_sample_size=max_sample_size, 
            log_scale=log_scale, 
            min_sample_size=min_sample_size, 
            absolute_scale=False
        )
        print(sample_sizes)
        print(dataset_dict)

        

        

       
        generator = torch.Generator().manual_seed(seed)
        pl.seed_everything(seed)
        
        
        

        for trainset in training_set:
            training_data = dataset_dict[trainset]
            
            for sample_size in sample_sizes:

                model = Classifier(num_class=num_classes)

                # Model Specific Checkpoint path. 
                model_checkpoint_path = os.path.join(checkpoint_path, current_dataset, experiment, trainset, str(sample_size), str(seed))
                if(os.path.exists(model_checkpoint_path) == False):
                    os.makedirs(model_checkpoint_path)

                # Initialize callbacks. 
                checkpoint_callback = ModelCheckpoint(
                    monitor='epoch_val_auroc',
                    dirpath=model_checkpoint_path,
                    filename='best-model'
                )
                early_stop_callback = EarlyStopping(
                    monitor="epoch_val_auroc", min_delta=0.00, patience=3, 
                    verbose=False, mode="max"
                )

                # Training, validation, test. split. 
                if(len(training_data) > sample_size):
                    train_subset, validation_subset = random_split(training_data, lengths = [sample_size, len(training_data) - sample_size], generator=generator)
                else:
                    train_split = int(len(training_data) * 0.8)
                    val_split = len(training_data) - train_split
                    train_subset, validation_subset = random_split(training_data, lengths = [train_split, val_split], generator=generator)
                
                #wrs = weighted_sampler(train_subset, mode="frequency") 
                
                trainloader = torch.utils.data.DataLoader(train_subset,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                #                                        sampler=wrs
                                                        )
                
                valloader = torch.utils.data.DataLoader(validation_subset,
                                                        batch_size=batch_size,
                                                        num_workers=num_workers,
                                                        pin_memory=True)
                
                
                # Initialize logger. 
                experiment_name = f"{experiment}__{current_dataset}__{trainset}__{sample_size}__{seed}"
                logger = TensorBoardLogger(save_dir=save_dir, name=experiment_name)
                
                # Find learning rate. 
                if(auto_lr_find == True):
                    if(os.path.exists(os.path.join(model_checkpoint_path, 'learning_rate')) == False):
                        os.makedirs(os.path.join(model_checkpoint_path, 'learning_rate'))

                    # Initialize Tuner. 
                    tuner = pl.tuner.Tuner(Trainer(default_root_dir=os.path.join(model_checkpoint_path, 'learning_rate')))
                    lr_finder = tuner.lr_find(model, trainloader, valloader)
                    fig = lr_finder.plot(suggest=True)
                    fig.savefig(os.path.join(model_checkpoint_path, 'learning_rate.png'))
                    print("Suggesting learning rate", lr_finder.suggestion())
                    model.learning_rate = lr_finder.suggestion()
                
                # Initializing trainer. 
                trainer = Trainer(
                    accelerator="gpu",
                    devices= num_gpus if torch.cuda.is_available() else None,  # limiting got iPython runs
                    num_nodes=num_nodes,
                    strategy=accelerator,
                    max_epochs=max_epochs,
                    callbacks=[
                        TQDMProgressBar(refresh_rate=refresh_rate), 
                        checkpoint_callback,
                        early_stop_callback
                    ],
                    #track_grad_norm=2,
                    gradient_clip_val=gradient_clip_val,
                    logger=logger
                )

                # Fit train loader, valloader. 
                trainer.fit(model, trainloader, valloader)

                test_results = list()
                for dataloader in test_dataloaders:
                    auroc_multiclass = trainer.test(model=model, dataloaders=dataloader)
                    test_results.append(auroc_multiclass[0])
                print(auroc_multiclass)
                output_file = pd.DataFrame((test_results))
                print(output_file)
                output_file['experiment'] = experiment_name
                output_file['seed'] = seed
                output_file['testsets'] = testing_set
                output_file['sample_size'] = len(train_subset)
                output_path = os.path.join(output_directory, experiment_name) + ".tsv"
                output_file.to_csv(output_path, sep="\t", index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for convergence sample size estimtation.')
    parser.add_argument("--config", required=True, type=str,
                        help="Config file path.")

    parser.add_argument("--seed", required=True, type=str,
                        help="Config file path.")
    config_kwargs = parser.parse_args()

    opt = vars(config_kwargs)
    config_kwargs = yaml.load(open(config_kwargs.config), Loader=yaml.FullLoader)
    opt.update(config_kwargs)
    config_kwargs = opt


    main(**(config_kwargs))