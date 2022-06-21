"""Data generator for CheXpert Dataset. 
"""
import os
import pandas as pd

data_dir = '/sc/arion/projects/mscic1/data/chexpert/'
working_dir = '/sc/arion/projects/mscic1/data/chexpert/CheXpert-v1.0'
train_path = os.path.join(working_dir, 'train.csv')
valid_path = os.path.join(working_dir, 'valid.csv')

def label_encoding(label_df:pd.DataFrame, labels:list):
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
            all_df['Sex']
        ]
        out_config = out_config + sexes
    if(age_range): 
        age_range = age_range.split(",")
        age_range = [int(age) for age in age_range]
        all_df = all_df[
            all_df['Age'].between(age_range[0], age_range[1])
        ]
        out_config = out_config = age_range
    
    # Encode labels
    labels = labels.split(",")
    out_config = out_config + labels
    all_df = label_encoding(
        label_df=all_df,
        labels=labels
    )

    # 
    out_path = os.path.join(root_dir, out_config) + ".tsv"
    all_df.to_csv(out_path, sep="\t", index=False)

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

        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            sexes="Male"
        )

        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2], tl_transforms=tl_transforms)
    
    if(dataset == "ages"):
        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            age_range="20,40"
        )

        metadata_path_1 = generate_metadata_file(
            root_dir=root_dir, 
            labels='Pneumonia,Pneumothorax,Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Lung Opacity,Pleural Effusion',
            orientations='AP',
            age_range="40,60"
        )

        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2], tl_transforms=tl_transforms)

