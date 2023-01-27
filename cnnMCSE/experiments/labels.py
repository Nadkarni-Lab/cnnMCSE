"""Experiments for label bias. 
"""
import random
import torch
import numpy as np
from random import Random
from torch.utils.data import TensorDataset, random_split, ConcatDataset, WeightedRandomSampler, Subset, RandomSampler

from scipy.stats import poisson
from cnnMCSE.dataloader import dataloader_helper

def labels_mnist(root_dir:str,
    tl_transforms:bool=False,
    bias_means:list=[1, 10],
    num_samples:list=[10000,10000] ,
    clip_val:float=0.05,
    start_seed:int=42,
    ):

    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)

    labels = mnist_trainset.targets
    bias_down_mean, bias_up_mean = bias_means
    samples_down, samples_up = num_samples

    # Generate poisson random variables. 
    rv_down = poisson(bias_down_mean)
    rv_up = poisson(bias_up_mean)

    # Weight labels by poisson distribution. 
    weights_down = rv_down.pmf(labels)
    #weights_up = 1 - weights_down
    weights_up   = rv_up.pmf(labels)

    # Clip distribution
    #weights_down = np.clip(weights_down, a_min=clip_val, a_max=1.0)
    #weights_up = np.clip(weights_up, a_min=clip_val, a_max=1.0)

    # get indices
    indices_down = list(WeightedRandomSampler(weights=weights_down, num_samples=samples_down, replacement=True, generator=generator))
    indices_up   = list(WeightedRandomSampler(weights=weights_up, num_samples=samples_up, replacement=True, generator=generator))

    # MNIST training - biased up and down samples. 
    mnist_trainset_down = Subset(dataset=mnist_trainset, indices=indices_down)
    mnist_trainset_up   = Subset(dataset=mnist_trainset, indices=indices_up)

    joint_trainset = ConcatDataset([mnist_trainset_down, mnist_trainset_up])
    joint_testset = mnist_testset

    print("Mnist trainset down", indices_down)
    print("Mnist trainset down", mnist_trainset_down)


    dataset_dict = {
        'subsample_1': mnist_trainset_down, 
        'subsample_2': mnist_trainset_up,
        'joint_trainset' : joint_trainset, 
        'testsample_1' : mnist_testset,
        'testsample_2' : mnist_testset,
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

def labels_fmnist(root_dir:str,
    tl_transforms:bool=False,
    bias_means:list=[1, 10],
    num_samples:list=[10000,10000] ,
    clip_val:float=0.05,
    start_seed:int=42,
    ):

    mnist_trainset, mnist_testset = dataloader_helper(dataset='FMNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)

    labels = mnist_trainset.targets
    bias_down_mean, bias_up_mean = bias_means
    samples_down, samples_up = num_samples

    # Generate poisson random variables. 
    rv_down = poisson(bias_down_mean)
    rv_up = poisson(bias_up_mean)

    # Weight labels by poisson distribution. 
    weights_down = rv_down.pmf(labels)
    #weights_up = 1 - weights_down
    weights_up   = rv_up.pmf(labels)

    # Clip distribution
    #weights_down = np.clip(weights_down, a_min=clip_val, a_max=1.0)
    #weights_up = np.clip(weights_up, a_min=clip_val, a_max=1.0)

    # get indices
    indices_down = list(WeightedRandomSampler(weights=weights_down, num_samples=samples_down, replacement=True, generator=generator))
    indices_up   = list(WeightedRandomSampler(weights=weights_up, num_samples=samples_up, replacement=True, generator=generator))

    # MNIST training - biased up and down samples. 
    mnist_trainset_down = Subset(dataset=mnist_trainset, indices=indices_down)
    mnist_trainset_up   = Subset(dataset=mnist_trainset, indices=indices_up)

    joint_trainset = ConcatDataset([mnist_trainset_down, mnist_trainset_up])
    joint_testset = mnist_testset

    print("Mnist trainset down", indices_down)
    print("Mnist trainset down", mnist_trainset_down)


    dataset_dict = {
        'subsample_1': mnist_trainset_down, 
        'subsample_2': mnist_trainset_up,
        'joint_trainset' : joint_trainset, 
        'testsample_1' : mnist_testset,
        'testsample_2' : mnist_testset,
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


def labels_cifar10(root_dir:str,
    tl_transforms:bool=False,
    bias_means:list=[1, 10],
    num_samples:list=[10000,10000] ,
    clip_val:float=0.05,
    start_seed:int=42,
    ):

    mnist_trainset, mnist_testset = dataloader_helper(dataset='CIFAR10', root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)

    labels = mnist_trainset.targets
    bias_down_mean, bias_up_mean = bias_means
    samples_down, samples_up = num_samples

    # Generate poisson random variables. 
    rv_down = poisson(bias_down_mean)
    rv_up = poisson(bias_up_mean)

    # Weight labels by poisson distribution. 
    weights_down = rv_down.pmf(labels)
    #weights_up = 1 - weights_down
    weights_up   = rv_up.pmf(labels)

    # Clip distribution
    #weights_down = np.clip(weights_down, a_min=clip_val, a_max=1.0)
    #weights_up = np.clip(weights_up, a_min=clip_val, a_max=1.0)

    # get indices
    indices_down = list(WeightedRandomSampler(weights=weights_down, num_samples=samples_down, replacement=True, generator=generator))
    indices_up   = list(WeightedRandomSampler(weights=weights_up, num_samples=samples_up, replacement=True, generator=generator))

    # MNIST training - biased up and down samples. 
    mnist_trainset_down = Subset(dataset=mnist_trainset, indices=indices_down)
    mnist_trainset_up   = Subset(dataset=mnist_trainset, indices=indices_up)

    joint_trainset = ConcatDataset([mnist_trainset_down, mnist_trainset_up])
    joint_testset = mnist_testset

    print("Mnist trainset down", indices_down)
    print("Mnist trainset down", mnist_trainset_down)


    dataset_dict = {
        'subsample_1': mnist_trainset_down, 
        'subsample_2': mnist_trainset_up,
        'joint_trainset' : joint_trainset, 
        'testsample_1' : mnist_testset,
        'testsample_2' : mnist_testset,
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


def mislabeled_MNIST(root_dir:str,
    tl_transforms:bool=False,
    bias_means:list=[1, 10],
    num_samples:list=[10000,10000] ,
    clip_val:float=0.05,
    start_seed:int=42,
    incorrect_label:int=1):
    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)
    np.random.seed(start_seed)
    labels = mnist_trainset.targets

    mislabel = 1
    percentage_mislabeled_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0]

    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    for percentage_mislabeled in percentage_mislabeled_list:
        
        # Initialize Training Data. 
        mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
        
        # Get Labels. 
        labels_list = labels.tolist()
        num_mislabeled = int(len(labels_list) * percentage_mislabeled)
        set_labels = set(labels_list)
        set_labels = list(set_labels)

        all_index = [i for i in range(len(labels_list)) if labels_list[i] == incorrect_label]
        num_mislabeled = int(len(all_index) * percentage_mislabeled)
        incorrect_indices = np.random.choice(all_index, size=num_mislabeled, replace=False)
        for incorrect_index in incorrect_indices:
            current_label = labels_list[incorrect_index]
            incorrect_labels = [set_label for set_label in set_labels if set_label != current_label]
            new_label = np.random.choice(incorrect_labels, size=1, replace=False)[0]
            labels_list[incorrect_index] = new_label
        mnist_trainset.targets = torch.Tensor(labels_list)
        dataset_dict[f'trainset__{percentage_mislabeled}'] = mnist_trainset
        dataset_dict[f'testset__{percentage_mislabeled}'] = mnist_testset
        dataset_dict['train_test_match'].append(
            [f'trainset__{percentage_mislabeled}',  f'testset__{percentage_mislabeled}']
        )
    
    return dataset_dict

def mislabeled_MNIST2(root_dir:str,
    tl_transforms:bool=False,
    bias_means:list=[1, 10],
    num_samples:list=[10000,10000] ,
    clip_val:float=0.05,
    start_seed:int=42,
    incorrect_label:int=1):

    # Set seed. 
    torch.manual_seed(start_seed)
    np.random.seed(start_seed)

    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    fake_trainset, fake_testset = dataloader_helper(dataset='FAKE', root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)
    #np.random.seed(start_seed)
    #labels = mnist_trainset.targets

    mislabel = 1
    percentage_mislabeled_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 0.95]

    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    for percentage_mislabeled in percentage_mislabeled_list:
        
        # Initialize Training Data. 
        mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
        fake_trainset, fake_testset = dataloader_helper(dataset='FAKE', root_dir=root_dir, tl_transforms=tl_transforms)

        #all_index = [i for i in range(len(labels_list)) if labels_list[i] == incorrect_label]
        num_mislabeled = int(len(mnist_trainset) * percentage_mislabeled)
        num_labeled = len(mnist_trainset) - num_mislabeled

        if(num_labeled != 0): 
            mnist_subset, _ = random_split(
                mnist_trainset, 
                lengths = [num_labeled, num_mislabeled], 
                generator=generator
            )
        if(num_mislabeled != 0): 
            fake_subset, _ = random_split(
                fake_trainset, 
                lengths = [num_mislabeled, num_labeled], 
                generator=generator
            )
        joint_trainset = ConcatDataset([mnist_subset, fake_subset])
       
        dataset_dict[f'trainset__{percentage_mislabeled}'] = joint_trainset
        dataset_dict[f'testset__{percentage_mislabeled}'] = mnist_testset
        dataset_dict['train_test_match'].append(
            [f'trainset__{percentage_mislabeled}',  f'testset__{percentage_mislabeled}']
        )
    
    return dataset_dict

def labels_generic(root_dir, dataset, tl_transforms, start_seed,
    bias_means:list=[1, 10],
    num_samples:list=[10000,10000] ,
    clip_val:float=0.05):
    synthetic_trainset, synthetic_testset = dataloader_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)

    labels = synthetic_trainset.targets
    bias_down_mean, bias_up_mean = bias_means
    samples_down, samples_up = num_samples

    # Generate poisson random variables. 
    rv_down = poisson(bias_down_mean)
    rv_up = poisson(bias_up_mean)

    # Weight labels by poisson distribution. 
    weights_down = rv_down.pmf(labels)
    #weights_up = 1 - weights_down
    weights_up   = rv_up.pmf(labels)

    # Clip distribution
    #weights_down = np.clip(weights_down, a_min=clip_val, a_max=1.0)
    #weights_up = np.clip(weights_up, a_min=clip_val, a_max=1.0)

    # get indices
    indices_down = list(WeightedRandomSampler(weights=weights_down, num_samples=samples_down, replacement=True, generator=generator))
    indices_up   = list(WeightedRandomSampler(weights=weights_up, num_samples=samples_up, replacement=True, generator=generator))

    # synthetic training - biased up and down samples. 
    synthetic_trainset_down = Subset(dataset=synthetic_trainset, indices=indices_down)
    synthetic_trainset_up   = Subset(dataset=synthetic_trainset, indices=indices_up)

    joint_trainset = ConcatDataset([synthetic_trainset_down, synthetic_trainset_up])
    joint_testset = synthetic_testset

    print(f"{dataset} trainset down", indices_down)
    print(f"{dataset} trainset down", synthetic_trainset_down)


    dataset_dict = {
        'subsample_1': synthetic_trainset_down, 
        'subsample_2': synthetic_trainset_up,
        'joint_trainset' : joint_trainset, 
        'testsample_1' : synthetic_testset,
        'testsample_2' : synthetic_testset,
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



def labels_helper(root_dir, dataset, tl_transforms, start_seed):

    if(dataset == "MNIST"):
        return labels_mnist(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    elif(dataset == "CIFAR10"):
        return labels_cifar10(root_dir=root_dir, tl_transforms=tl_transforms)
    elif(dataset == "misMNIST"):
        return mislabeled_MNIST(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    elif(dataset == "misMNIST2"):
        return mislabeled_MNIST2(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    elif(dataset == "FMNIST"):
        return labels_fmnist(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    else:
        return labels_generic(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed, dataset=dataset)