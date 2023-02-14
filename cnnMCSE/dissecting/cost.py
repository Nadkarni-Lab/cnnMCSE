import numpy as np
import pandas as pd 
import torch

from os.path import exists
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader


from cnnMCSE.dissecting.util.features import *
from cnnMCSE.dissecting.util.main import *
from cnnMCSE.dissecting.util.model import *
from cnnMCSE.dissecting.util.util import *

def label_encoding(y_df:pd.DataFrame, outcome:str, threshold:float=None, thresholds:list=None):
    """Encode labels in a metadata file. 

    Args:
        metadata_df (pd.DataFrame): Metadata data frame generated synthetically. 
        outcome (str): Outcome value. 
        threshold (float): Threshold to classify. 
    """
    if(threshold):
        y_df['target'] = (y_df[outcome] > y_df[outcome].quantile(q=threshold)).apply(int)
        y_df['target_map'] = np.where(y_df['target'] == 1, 'high_risk', 'low_risk')
    
    if(thresholds):
        y_df['target'] = 0
        for index, threshold in enumerate(thresholds):
            y_df.loc[(y_df[outcome] > y_df[outcome].quantile(q=threshold)), ['target']] = index 
        y_df['target_map'] = np.where(y_df['target'] > 5, 'high_risk', 'low_risk')

    return y_df


def generate_metadata_file(metadata_path:str, root_dir:str, demographics:str, outcome:str, threshold:float=None, thresholds:list=None):
    """Generate a metadata file.

    Args:
        metadata_path (str): Path to metadata file. 
        root_dir (str): Root directory to save all the files. 
        demographics (str): Which demographic to pick. 
        outcome (str): Outcome to use. 
        threshold (float, optional): Which threshold to focus on. Defaults to 0.5.

    Returns:
        str: Path to metadatas frame. 
    """
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.reset_index()
    out_config = list()
    
    # Get inputs and outputs. 
    metadata_df, x_column_names, Y_predictors = get_Y_x_df(metadata_df, verbose=True)

    if(demographics):
        if(demographics == "white"):
            metadata_df = metadata_df[
                metadata_df['dem_race_black'] == 0
            ]
        if(demographics == "black"):
            metadata_df = metadata_df[
                metadata_df['dem_race_black'] == 1
            ]
        out_config.append(demographics)
    
    # Scale inputs
    scaler = MinMaxScaler()
    X_vars = metadata_df[x_column_names]
    X_vars = scaler.fit_transform(X_vars.to_numpy())
    X_df = pd.DataFrame(X_vars, columns=x_column_names)

    # Label encoding. 
    y_df = metadata_df[Y_predictors]
    y_df = label_encoding(y_df, outcome, threshold=threshold, thresholds=thresholds)
    y_df = y_df.drop(Y_predictors, axis=1)

    # Merge X and y dataframes. 
    X_df['target'] = list(y_df['target'])
    metadata_df = X_df

    out_config.append(outcome)
    if(threshold):
        out_config.append(threshold)
    if(thresholds):
        thresholds_str = [str(thresh) for thresh in thresholds]
        thresholds_str = '_'.join(thresholds_str)
        out_config.append(thresholds_str)
    out_config = [str(config) for config in out_config]
    
    out_metadata_filename = '_'.join(out_config)
    out_metadata_filename = out_metadata_filename + '.tsv'
    out_metadata_filename = out_metadata_filename.replace("/", "_")
    out_metadata_path = os.path.join(root_dir, out_metadata_filename)

    if(exists(out_metadata_path) == False): 
        metadata_df.to_csv(out_metadata_path, sep="\t", index=False)
    return out_metadata_path

def generate_dataloaders(metadata_paths:list, tags:list,  start_seed:float, split_ratio:float=0.7):
    """Generate a set of dataloaders by metadta paths, tags, start seed and split ratios. 

    Args:
        metadata_paths (list): Path to list of experiments that need to run. 
        tags (list): List of tags associated with each metadata file. 
        start_seed (float): Start seed to ensure replicability.
        split_ratio (float, optional): Train-test-split ratio. Defaults to 0.7.

    Returns:
        _type_: _description_
    """
    # Generate seeds. 
    generator = torch.Generator().manual_seed(start_seed)

    # Initialize dataset dictionary
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()

    for path, tag in zip(metadata_paths, tags):
        # Generate dataset
        current_dataset     = DissectingDataset(metadata_path = path)

        # Generate train-test-split ratio. 
        len_current_dataset = len(current_dataset)
        len_trainset        = int(len_current_dataset * split_ratio)
        len_testset         = len_current_dataset - len_trainset
        trainset, testset = random_split(current_dataset, [len_trainset, len_testset], generator=generator)

        # Generate dataset_dict
        dataset_dict[f'{tag}_train'] =  trainset
        dataset_dict[f'{tag}_test'] =  testset
        dataset_dict['train_test_match'].append([f'{tag}_train', f'{tag}_test'])
    
    return dataset_dict



class DissectingDataset(Dataset):
    """Dataset for Dissecting Datsaet bias

    Args:
        Dataset: Torch dataset type. 
    """
    def __init__(self, metadata_path, transform=None, target_transform=None):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path, sep="\t")
        self.targets = torch.from_numpy(np.array(self.metadata_df['target']))
        self.inputs = self.metadata_df.drop(['target'], axis=1)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.metadata_df.shape[0]
    
    def __getitem__(self, idx):
        predictors = torch.Tensor(self.inputs.iloc[idx])
        outcomes = self.targets[idx]

        if(self.transform):
            predictors = self.transform(predictors)
        
        if(self.target_transform):
            outcomes = self.target_transform(outcomes)
        
        return predictors, outcomes


def dissecting_helper(root_dir:str, dataset:str, start_seed:int):
    if(dataset == "outcomes"):
        metadata_path_cost = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics=None,
            outcome = 'log_cost_t',
            threshold=0.5
        )

        metadata_path_gagne = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics=None,
            outcome = 'gagne_sum_t',
            threshold=0.5
        )

        metadata_path_avoidable = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics=None,
            outcome = 'log_cost_avoidable_t',
            threshold=0.5
        )
        
        # Create lists of metadata paths and tags. 
        metadata_paths = [metadata_path_cost, metadata_path_gagne, metadata_path_avoidable]
        tags = ['cost', 'gagne', 'avoidable']

        return generate_dataloaders(metadata_paths=metadata_paths, tags=tags, start_seed=start_seed)
    
    if(dataset == "ethnicity"):
        metadata_path_cost_white = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics="white",
            outcome = 'log_cost_t',
            threshold=0.5
        )

        metadata_path_gagne_white = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics="white",
            outcome = 'gagne_sum_t',
            threshold=0.5
        )

        metadata_path_avoidable_white = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics="white",
            outcome = 'log_cost_avoidable_t',
            threshold=0.5
        )

        metadata_path_cost_black = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics="black",
            outcome = 'log_cost_t',
            threshold=0.5
        )

        metadata_path_gagne_black = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics="black",
            outcome = 'gagne_sum_t',
            threshold=0.5
        )

        metadata_path_avoidable_black = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics="black",
            outcome = 'log_cost_avoidable_t',
            threshold=0.5
        )
        
        # Create lists of metadata paths and tags. 
        metadata_paths = [
            metadata_path_cost_white, metadata_path_gagne_white, metadata_path_avoidable_white,
            metadata_path_cost_black, metadata_path_gagne_black, metadata_path_avoidable_black
        ]
        tags = [
            'cost_white', 'gagne_white', 'avoidable_white',
            'cost_black', 'gagne_black', 'avoidable_black'
        ]

        return generate_dataloaders(metadata_paths=metadata_paths, tags=tags, start_seed=start_seed)
    
    if(dataset == "stratification"):
        metadata_path_cost = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics=None,
            outcome = 'log_cost_t',
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        metadata_path_gagne = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics=None,
            outcome = 'gagne_sum_t',
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        metadata_path_avoidable = generate_metadata_file(
            root_dir=root_dir, 
            metadata_path= '/sc/arion/projects/EHR_ML/gulamf01/dissecting-bias/data/data_new.csv',
            demographics=None,
            outcome = 'log_cost_avoidable_t',
            thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Create lists of metadata paths and tags. 
        metadata_paths = [metadata_path_cost, metadata_path_gagne, metadata_path_avoidable]
        tags = ['cost', 'gagne', 'avoidable']

        return generate_dataloaders(metadata_paths=metadata_paths, tags=tags, start_seed=start_seed)