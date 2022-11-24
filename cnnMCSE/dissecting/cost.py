import numpy as np
import pandas as pd 
import torch
from torch.utils.data import Dataset, random_split, ConcatDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def label_encoding(y_df:pd.DataFrame, outcome:str, threshold:float):
    """Encode labels in a metadata file. 

    Args:
        metadata_df (pd.DataFrame): _description_
        outcome (str): Outcome value. 
        threshold (float): Threshold to classify. 
    """
    y_df['target'] = (y_df[outcome] > y_df[outcome].quantile(q=threshold)).apply(float)
    y_df['target_map'] = np.where(y_df['target'] == 1, 'high_risk', 'low_risk')
    return y_df


def generate_metadata_file(metadata_path:str, root_dir:str, demographics:str, outcome:str, threshold:float=0.5):
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
    y_df = label_encoding(y_df, outcome, threshold)
    y_df = y_df.drop(Y_predictors, axis=1)

    # Merge X and y dataframes. 
    X_df['target'] = list(y_df['target'])
    metadata_df = X_df

    out_config.append(outcome)
    out_config.append(threshold)
    
    out_metadata_filename = '_'.join(out_config)
    out_metadata_filename = out_metadata_filename + '.tsv'
    out_metadata_filename = out_metadata_filename.replace("/", "_")


    out_metadata_filename = out_config + '.tsv'
    out_metadata_path = os.path.join(root_dir, out_metadata_filename)

    if(exists(out_metadata_path) == False): 
        metadata_df.to_csv(out_metadata_path, sep="\t", index=False)
    return out_metadata_path

def generate_dataloaders(metadata_paths:list):

    # Generate seeds. 
    generator_1 = torch.Generator().manual_seed(start_seed)
    generator_2 = torch.Generator().manual_seed(start_seed+1)
    metadata_path_1, metadata_path_2 = metadata_paths


class DissectingDataset(Dataset):
    """Class for Datasets for the Dissecting Dataset Bias Paper. 
    """
    def __init__(self, metadata_path, transform=None, target_transform=None):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path, sep="\t")
        self.targets = list(self.metadata_df['target'])
        self.inputs = self.metadata_df.drop(['target'], axis=1)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.metadata_df.shape[0]
    
    def __getitem__(self, idx):
        predictors = torch.Tensor(self.inputs.iloc[idx])
        outcomes = torch.Tensor([self.targets[idx]])

        if(self.transform):
            predictors = self.transform(predictors)
        
        if(self.target_transform):
            outcomes = self.target_transform(outcomes)
        
        return predictors, outcomes


