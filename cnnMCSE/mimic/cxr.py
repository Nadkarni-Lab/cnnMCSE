"""File to generate data loaders for experimentation with the MIMIC-CXR database. 
"""
import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
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


def generate_metadata_file(root_dir:str, labels:str, demographics:str, orientations:str, insurances:str=None):
    
    # read in the records
    cxr_record_df = pd.read_csv(record_list)
    metadata_df = pd.read_csv(metadata)

    # split the lists. 
    demographics = demographics.split(",")
    labels = labels.split(",")
    orientations = orientations.split(",")
    insurances = insurances.split(",")

    # create outpath:
    out_config = demographics + labels + orientations + insurances
    out_metadata_filename = '_'.join(out_config)
    out_metadata_filename = out_metadata_filename + '.tsv'
    out_metadata_filename = out_metadata_filename.replace("/", "_")
    out_metadata_path = os.path.join(root_dir, out_metadata_filename)
    print(out_metadata_filename)
    print(out_metadata_path)


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
    ethnicity_df = mimic_admission_df[['subject_id', 'ethnicity', 'insurance']].drop_duplicates()
    cxr_record_df = pd.merge(cxr_record_df, ethnicity_df, on='subject_id')
    cxr_record_df = cxr_record_df[
        cxr_record_df['ethnicity'].isin(demographics)
    ]

    if(insurances):
        cxr_record_df = cxr_record_df[
            cxr_record_df['insurance'].isin(insurances)
        ]
        print("Length", len(cxr_record_df))



    # select relevant labels. 
    labels_df = pd.read_csv(chexpert)
    labels_df = labels_df[['subject_id', 'study_id'] + labels]
    encoded_df = label_encoding(labels_df, labels)
    cxr_record_df = pd.merge(cxr_record_df, encoded_df, on = ['subject_id', 'study_id'])

    # export dataframe. 
    cxr_record_df.to_csv(out_metadata_path, sep="\t", index=False)
    return out_metadata_path
    

def generate_dataloaders(
        metadata_path:str=None, 
        metadata_paths:str=None,
        tl_transforms:bool=False,
        split_ratio:float=0.8, 
        start_seed:int=42
    ):
    generator_1 = torch.Generator().manual_seed(start_seed)
    generator_2 = torch.Generator().manual_seed(start_seed+1)

    metadata_path_1, metadata_path_2 = metadata_paths
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

    dataset_1 = MimicCXRDataset(
        metadata_path=metadata_path_1,
        transform=transform
    )
    len_dataset_1 = len(dataset_1)
    len_trainset_1 = int(len_dataset_1 * split_ratio)
    len_testset_1 = len_dataset_1 - len_trainset_1
    trainset_1, testset_1 = random_split(dataset_1, [len_trainset_1, len_testset_1], generator=generator_1)

    dataset_2 = MimicCXRDataset(
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

def ethnicity_experiments(root_dir:str):
    pass


def mimic_helper(dataset, root_dir):
    if(dataset == "test"):
        generate_metadata_file(
            out_path=root_dir, 
            labels='Pneumonia,Pneumothorax',
            demographics='WHITE',
            orientations='postero-anterior'

        )
        return generate_dataloaders(metadata_path=root_dir)
    
    if(dataset == "ethnicity"):
        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax',
            demographics='WHITE',
            orientations='postero-anterior'
        )
        metadata_path_2 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax',
            demographics='BLACK/AFRICAN AMERICAN',
            orientations='postero-anterior'
        )
        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2])
    
    if(dataset == "medicare"):
        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax',
            demographics='WHITE',
            orientations='postero-anterior',
            insurances='Medicare'
        )
        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax',
            demographics='WHITE',
            orientations='postero-anterior',
            insurances='Other'
        )
        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2])



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


