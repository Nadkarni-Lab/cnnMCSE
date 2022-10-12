"""File to help with experimentation.
"""

import numpy as np
from torch.utils.data import Subset
from cnnMCSE.dataloader import synthetic_dataset, get_labels


def informative_synthetic(max_sample_size:int = 6000, n_informative_list:list=[2,4,8,16,32,64,128,256,512], n_features:int=784, n_classes:int=10): 
    """Generate dataset with varying number of informative features. 

    Args:
        max_sample_size (int, optional): _description_. Defaults to 6000.
        n_informative_list (list, optional): _description_. Defaults to [2,4,8,16,32,64,128,256,512].
        n_features (int, optional): _description_. Defaults to 784.
        n_classes (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    n_informative_list=[20, 40, 60, 80, 100, 150, 200, 400, 600]
    print(n_informative_list)
    for n_dim in n_informative_list:
        subsample_name = f'{n_dim}'
        print(max_sample_size)
        print(n_dim)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_dim, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

        dataset_dict[f'{n_dim}_trainset'] = trainset
        dataset_dict[f'{n_dim}_testset'] = testset
        dataset_dict['train_test_match'].append(
            [f'{n_dim}_trainset',  f'{n_dim}_testset']
        )
    
    return dataset_dict

def classes_synthetic(max_sample_size:int = 6000, n_informative:int=128, n_features:int=784, n_classes_list:list=[2, 4, 8, 10, 15, 20, 40]): 
    """Generate synthetic dataset with varying number of classes. 

    Args:
        max_sample_size (int, optional): _description_. Defaults to 6000.
        n_informative (int, optional): _description_. Defaults to 128.
        n_features (int, optional): _description_. Defaults to 784.
        n_classes_list (list, optional): _description_. Defaults to [2, 4, 8, 10, 15, 20, 40].

    Returns:
        _type_: _description_
    """
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    n_classes_list=[2, 4, 6, 8, 10]
    print(n_classes_list)
    for n_classes in n_classes_list:
        subsample_name = f'{n_classes}'
        print(max_sample_size)
        print(n_classes)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

        dataset_dict[f'{n_classes}_trainset'] = trainset
        dataset_dict[f'{n_classes}_testset'] = testset
        dataset_dict['train_test_match'].append(
            [f'{n_classes}_trainset',  f'{n_classes}_testset']
        )
    
    return dataset_dict



def flip_synthetic(max_sample_size:int = 6000, n_informative:int=256, n_features:int=784, n_classes:int=2): 
    """Generate synthetic dataset with varying number of % correctly labeled. 

    Args:
        max_sample_size (int, optional): _description_. Defaults to 6000.
        n_informative (int, optional): _description_. Defaults to 256.
        n_features (int, optional): _description_. Defaults to 784.
        n_classes (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    flip_y_list = [0.001, 0.01, 0.02, 0.05, 0.10, 0.20,0.30, 0.40, 0.50]
    for flip_y in flip_y_list:
        subsample_name = f'{flip_y}'
        print(max_sample_size)
        print(flip_y)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes,flip_y=flip_y, train_test_split=5000, seed=42)

        dataset_dict[f'trainset__{flip_y}'] = trainset
        dataset_dict[f'testset__{flip_y}'] = testset
        dataset_dict['train_test_match'].append(
            [f'trainset__{flip_y}',  f'testset__{flip_y}']
        )
    
    return dataset_dict


def dim_synthetic(max_sample_size:int = 6000, n_informative:int=128, n_features:int=784, n_classes:int=10): 
    """Generate synthetic dataset with varying number of features. 

    Args:
        max_sample_size (int, optional): _description_. Defaults to 6000.
        n_informative (int, optional): _description_. Defaults to 128.
        n_features (int, optional): _description_. Defaults to 784.
        n_classes (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()

    trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

    dataset_dict[f'trainset__{n_features}'] = trainset
    dataset_dict[f'testset__{n_features}'] = testset
    dataset_dict['train_test_match'].append(
        [f'trainset__{n_features}',  f'testset__{n_features}']
    )

    return dataset_dict

def sampling_bias(max_sample_size:int = 6000, n_informative:int=512, n_features:int=784, n_classes:int=2):
    """Method to generate a dataset with sampling bias. 

    Args:
        max_sample_size (int, optional): Maximum sample size of the dataset. Defaults to 6000.
        n_informative (int, optional): Number of informative features in the dataset. Defaults to 784.
        n_features (int, optional): Number of total features. Defaults to 784.
        n_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
        _type_: _description_
    """
    
    sampling_ratios = [0.01, 0.02, 0.05, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50]

    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    for index, ratio in enumerate(sampling_ratios):
        print("n features ", n_features)
        # Generate and seed the data. 
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=index)

        # get a list of targets
        targets = get_labels(trainset)

        # get a list of true and false indices
        indices_true = [i for i in range(len(targets)) if targets[i] == 1]
        indices_false = [i for i in range(len(targets)) if targets[i] != 1]

        # identify the number of true and false indices required. 
        num_true = int(ratio * len(targets))
        num_false = len(targets) - num_true

        # subset the indices. 
        #np.random.seed(index)
        subset_indices_true = np.random.choice(a = indices_true, size = num_true, replace=True)
        subset_indices_false = np.random.choice(a = indices_false, size = num_false, replace=True)
        subset_indices_total = list(subset_indices_true) + list(subset_indices_false)
        subset_train = Subset(dataset=trainset, indices=subset_indices_total)

        dataset_dict[f'trainset__{ratio}'] = subset_train
        dataset_dict[f'testset__{ratio}'] = testset
        dataset_dict['train_test_match'].append(
            [f'trainset__{ratio}',  f'testset__{ratio}']
        )
    
    return dataset_dict 




        






def synthetic_helper(root_dir, dataset, input_dim:int=None, tl_transforms:bool=False):
    if(dataset == "informative"):
        return informative_synthetic()
    if(dataset == "classes"):
        return classes_synthetic()
    if(dataset == "flip"):
        return flip_synthetic()
    if(dataset == "features"):
        return dim_synthetic(n_features=input_dim)
    if(dataset == "sampling_bias"):
        return sampling_bias(n_features=input_dim)

    else:
        return None

