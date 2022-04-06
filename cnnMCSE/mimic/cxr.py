"""File to generate data loaders for experimentation with the MIMIC-CXR database. 
"""
import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms



JPG_DIR = '/sc/arion/projects/mscic1/data/mimic-cxr-jpg/physionet.org/files/mimic-cxr-jpg/2.0.0'
RECORD_DIR = '/sc/arion/projects/mscic1/data/mimc-cxr/physionet.org/files/mimic-cxr/2.0.0'
DEM_DIR = '/sc/arion/projects/mscic1/data/mimic-iv-v1/physionet.org/files/mimiciv/1.0/core'
metadata = os.path.join(JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv.gz')
chexpert = os.path.join(JPG_DIR, 'mimic-cxr-2.0.0-chexpert.csv.gz')
negbio = os.path.join(JPG_DIR, 'mimic-cxr-2.0.0-negbio.csv.gz')
split = os.path.join(JPG_DIR, 'mimic-cxr-2.0.0-split.csv.gz')
record_list = os.path.join(RECORD_DIR, 'cxr-record-list.csv.gz')
study_list = os.path.join(RECORD_DIR, 'cxr-study-list.csv.gz')
mimic_pt_path = os.path.join(DEM_DIR, 'patients.csv.gz')
mimic_transfer_path = os.path.join(DEM_DIR, 'transfers.csv.gz')
mimic_admissions_path = os.path.join(DEM_DIR, 'admissions.csv.gz')

def label_encoding(labels_df:pd.DataFrame, labels:list)->pd.DataFrame:
    """_Method to label encode. 

    Args:
        labels_df (pd.DataFrame): Pandas dataframe with labels. 
        labels (list): List of labels to encode. 

    Returns:
        pd.DataFrame: Pandas dataframe with encoded labels. 
    """

    # initialize list of targets and relevant mappings. 
    targets = list()
    targets_mapping = list()

    # iterate across the list. 
    for index, row in labels_df.iterrows():
        current_target = 0
        current_label = 'None'
        for val, label in enumerate(labels):
            if(row[label] == 1):
                current_target = val+1
                current_label = label
        targets_mapping.append(current_label)
        targets.append(current_target)
    
    # append to list. 
    labels_df['target'] = targets
    labels_df['target_map'] = targets_mapping
    return labels_df


def generate_metadata_file(out_path:str, labels:str, demographics:str, orientations:str):
    
    # read in the records
    cxr_record_df = pd.read_csv(record_list)
    metadata_df = pd.read_csv(metadata)

    # split the lists. 
    demographics = demographics.split(",")
    labels = labels.split(",")
    orientations = orientations.split(",")

    # establish paths
    cxr_record_df['path'] = cxr_record_df['path'].str.replace(".dcm", ".jpg")
    cxr_record_df['path'] = JPG_DIR + '/' + cxr_record_df['path'] 

    # select relevant X-ray orientation
    orientation_df = metadata_df[['subject_id', 'study_id', 'dicom_id', 'ViewCodeSequence_CodeMeaning']]    
    cxr_record_df = pd.merge(cxr_record_df, orientation_df, on = ['subject_id', 'study_id', 'dicom_id'])
    cxr_record_df = cxr_record_df[
        cxr_record_df['ViewCodeSequence_CodeMeaning'].isin(orientations)
    ]

    # select relevant subgroups. 
    mimic_admission_df = pd.read_csv(mimic_admissions_path)
    ethnicity_df = mimic_admission_df[['subject_id', 'ethnicity']].drop_duplicates()
    cxr_record_df = pd.merge(cxr_record_df, ethnicity_df, on='subject_id')
    cxr_record_df = cxr_record_df[
        cxr_record_df['ethnicity'].isin(demographics)
    ]

    # select relevant labels. 
    labels_df = pd.read_csv(chexpert)
    labels_df = labels_df[['subject_id', 'study_id'] + labels]
    encoded_df = label_encoding(labels_df, labels)
    cxr_record_df = pd.merge(cxr_record_df, encoded_df, on = ['subject_id', 'study_id'])

    # export dataframe. 
    cxr_record_df.to_csv(out_path, sep="\t", index=False)
    

def generate_dataloaders(
        metadata_path:str, 
        tl_transforms:bool=False,
        split_ratio:float=0.8, 
        start_seed:int=42
    ):
    generator_1 = torch.Generator().manual_seed(start_seed)
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

    dataset = MimicCXRDataset(
        metadata_path=metadata_path,
        transform=transform
    )
    len_dataset = len(dataset)
    len_trainset = int(len_dataset * split_ratio)
    len_testset = len_dataset - len_trainset
    trainset_1, testset_1 = random_split(dataset, [len_trainset, len_testset], generator=generator_1)
    # trainset_2, testset_2 = random_split(dataset, [len_dataset*split_ratio, len_dataset*(1-split_ratio)], generator=generator_2)

    dataset_dict = {
        'subsample_1': trainset_1, 
        'testsample_1': testset_1, 
        'train_test_match': [
            ['subsample_1', 'testsample_1']
        ]
    }
    return dataset_dict


def mimic_helper(dataset, root_dir):
    if(dataset == "test"):
        generate_metadata_file(
            out_path=root_dir, 
            labels='Pneumonia,Pneumothorax',
            demographics='WHITE',
            orientations='postero-anterior'

        )
        return generate_dataloaders(metadata_path=root_dir)



class MimicCXRDataset(Dataset):
    """Dataset for mimic CXR. 
    """

    def __init__(self, metadata_path, transform=None, target_transform=None):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path, sep="\t")
        self.transform = transform
        self.target_transform = target_transform
    
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


