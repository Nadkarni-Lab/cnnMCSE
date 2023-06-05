"""Data generator for CheXpert Dataset. 
"""
import os
import pandas as pd

from os.path import exists
from PIL import Image

import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
from torchvision import transforms

data_dir = '/sc/arion/projects/mscic1/data/chexpert/'
working_dir = '/sc/arion/projects/mscic1/data/chexpert/CheXpert-v1.0'
train_path = os.path.join(working_dir, 'train.csv')
valid_path = os.path.join(working_dir, 'valid.csv')

def label_encoding(label_df:pd.DataFrame, labels_list:list):
    targets_mapping = list()
    targets = list()
    for index, row in label_df.iterrows():
        current_target = 0
        current_label = 'None'
        for val, label in enumerate(labels_list):
            if(row[label] == 1):
                current_target = val+1
                current_label = label
        targets_mapping.append(current_label)
        targets.append(current_target)

    label_df['target'] = targets
    label_df['target_map'] = targets_mapping
    return label_df

def generate_metadata_file(root_dir:str, orientations:str, labels:str, sexes:str=None, age_range:str=None)->str:
    
    # initialize training and validation DF
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    all_df = pd.concat([train_df, valid_df])

    all_df['path'] = data_dir + all_df['Path']


    # Initialize Out config file. 
    out_config = ['chexpert']
    
    # Get relevant orientations
    orientations = orientations.split(",")
    all_df = all_df[
        all_df['AP/PA'].isin(orientations)
    ]
    out_config = out_config + orientations

    
    # Get relevant sexes. 
    if(sexes):  
        sexes = sexes.split(",")
        all_df = all_df[
            all_df['Sex'].isin(sexes)
        ]
        out_config = out_config + sexes
    if(age_range): 
        age_range = age_range.split(",")
        ages = [int(age) for age in age_range]
        min_age, max_age = ages
        all_df = all_df[
            all_df['Age'].between(min_age, max_age)
        ]
        out_config = out_config = age_range
    
    # Encode labels
    labels = labels.split(",")
    out_config = out_config + labels
    all_df = label_encoding(
        label_df=all_df,
        labels_list=labels
    )


    # Initialize output directory
    out_config = "_".join(out_config)
    out_path = os.path.join(root_dir, out_config) + ".tsv"
    all_df.to_csv(out_path, sep="\t", index=False)
    return out_path

def generate_dataloaders(
        metadata_path:str=None, 
        metadata_paths:str=None,
        tl_transforms:bool=False,
        split_ratio:float=0.8, 
        start_seed:int=42
    ):
    print('TL transforms', tl_transforms)
    generator_1 = torch.Generator().manual_seed(start_seed)
    generator_2 = torch.Generator().manual_seed(start_seed+1)
    if(metadata_paths):
        metadata_path_1, metadata_path_2 = metadata_paths
    else:
        metadata_path_1 = metadata_path
        metadata_path_2 = metadata_path
    # generator_2 = torch.Generator().manual_seed(start_seed+1)

    if(tl_transforms == False):
        to_grayscale = transforms.Grayscale()
        resize = transforms.Resize((28,28))
        normalize = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([
            resize,
            to_grayscale,
            transforms.ToTensor(),
            normalize
        ])
    elif(tl_transforms == True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset_1 = ChexpertDataset(
        metadata_path=metadata_path_1,
        transform=transform
    )
    len_dataset_1 = len(dataset_1)
    len_trainset_1 = int(len_dataset_1 * split_ratio)
    len_testset_1 = len_dataset_1 - len_trainset_1
    trainset_1, testset_1 = random_split(dataset_1, [len_trainset_1, len_testset_1], generator=generator_1)

    dataset_2 = ChexpertDataset(
        metadata_path=metadata_path_2,
        transform=transform
    )
    len_dataset_2 = len(dataset_2)
    len_trainset_2 = int(len_dataset_2 * split_ratio)
    len_testset_2 = len_dataset_2 - len_trainset_2
    trainset_2, testset_2 = random_split(dataset_2, [len_trainset_2, len_testset_2], generator=generator_2)
    # trainset_2, testset_2 = random_split(dataset, [len_dataset*split_ratio, len_dataset*(1-split_ratio)], generator=generator_2)
    joint_trainset = ConcatDataset([trainset_1, trainset_2])
    joint_testset = ConcatDataset([testset_1, testset_2])

    dataset_dict = {
        'subsample_1': trainset_1, 
        'testsample_1': testset_1, 
        'subsample_2' : trainset_2,
        'testsample_2': testset_2,
        'joint_trainset' : joint_trainset,
        'joint_testset' : joint_testset,
        'train_test_match': [
            ['subsample_1', 'testsample_1'], 
            ['subsample_2', 'testsample_2'], 
            ['joint_trainset', 'joint_testset'], 
            ['joint_trainset', 'testsample_1'],
            ['joint_trainset', 'testsample_2']
        ]
    }
    return dataset_dict

def chexpert_helper(dataset, root_dir, tl_transforms:bool=False):
    if(dataset == "test"):
        metadata_path = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP'

        )
        return generate_dataloaders(metadata_path=metadata_path, tl_transforms=tl_transforms)
    
    if(dataset == "test"):
        metadata_path = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP'

        )
        return generate_dataloaders(metadata_path=metadata_path, tl_transforms=tl_transforms)
    
    if(dataset == "gender"):
        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            sexes="Female"
        )

        metadata_path_2 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            sexes="Male"
        )

        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2], tl_transforms=tl_transforms)
    
    if(dataset == "age"):
        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            age_range="20,40"
        )

        metadata_path_2 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            age_range="40,60"
        )

        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2], tl_transforms=tl_transforms)
    


class ChexpertDataset(Dataset):
    """Dataset for mimic CXR. 
    """

    def __init__(self, metadata_path, transform=None, target_transform=None):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path, sep="\t")
        self.transform = transform
        self.target_transform = target_transform
        self.targets = list(self.metadata_df['target'])
    
    def __len__(self):
        return self.metadata_df.shape[0]
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        path = row['path']
        img = Image.open(path)
        target = row['target']

        if(self.transform):
            img = self.transform(img)
        
        if(self.target_transform):
            target = self.target_transform(target)
        
        return img, target

