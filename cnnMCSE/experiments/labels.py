"""Experiments for label bias. 
"""
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split, ConcatDataset, WeightedRandomSampler, Subset

from scipy.stats import poisson
from cnnMCSE.dataloader import dataloader_helper

def labels_mnist(root_dir:str,
    tl_transforms:bool=False,
    bias_means:list=[3, 7],
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
    weights_up   = rv_up.pmf(labels)

    # Clip distribution
    weights_down = np.clip(weights_down, a_min=clip_val, a_max=1.0)
    weights_up = np.clip(weights_up, a_min=clip_val, a_max=1.0)

    # get indices
    indices_down = list(WeightedRandomSampler(weights=weights_down, num_samples=samples_down, replacement=False, generator=generator))
    indices_up   = list(WeightedRandomSampler(weights=weights_up, num_samples=samples_up, replacement=False, generator=generator))

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


def labels_helper(root_dir, dataset):

    if(dataset == "MNIST"):
        return labels_mnist(root_dir=root_dir)
    else:
        return None