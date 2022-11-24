"""Experiments for label bias. 
"""
from random import Random
import torch
import numpy as np
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
    start_seed:int=42):
    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    generator = torch.Generator().manual_seed(start_seed)
    labels = mnist_trainset.targets

    mislabel = 1
    percentage_mislabeled_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1.0]

    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    for percentage_mislabeled in percentage_mislabeled_list:
        mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
        correct_indices = [i for i in range(len(labels)) if i == mislabel]
        incorrect_indices = [i for i in range(len(labels)) if i != mislabel]
        num_incorrect = int(len(correct_indices) * percentage_mislabeled) + 1
        print(num_incorrect)
        class_I_indices = list(RandomSampler(data_source=incorrect_indices, replacement=True, num_samples=num_incorrect)) + correct_indices
        adjusted_labels = list()
        for index, label in enumerate(labels):
            if index not in class_I_indices:
                adjusted_labels.append(label)
            else:
                adjusted_labels.append(1)
        
        mnist_trainset.targets = adjusted_labels
        dataset_dict[f'trainset__{percentage_mislabeled}'] = mnist_trainset
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
    elif(dataset == "FMNIST"):
        return labels_fmnist(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    else:
        return labels_generic(root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed, dataset=dataset)