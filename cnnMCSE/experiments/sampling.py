"""Experiments for sampling bias. 
"""
from ntpath import join
import torch
from torch.utils.data import TensorDataset, random_split, ConcatDataset

from cnnMCSE.dataloader import dataloader_helper


def bias_synthetic():

    pass


def bias_mnist(root_dir, 
    sample_sizes:list=[50000, 10000], 
    tl_transforms:bool=False,
    start_seed:int=42):

    generator = torch.Generator().manual_seed(start_seed)
    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    fmnist_trainset, fmnist_testset = dataloader_helper(dataset='FMNIST', root_dir=root_dir, tl_transforms=tl_transforms)

    mnist_length, fmnist_length = sample_sizes

    mnist_trainset_subsample, _ = random_split(
        mnist_trainset, 
        lengths = [mnist_length, len(mnist_trainset) - mnist_length], 
        generator=generator
    )

    fmnist_trainset_subsample, _ = random_split(
        fmnist_trainset, 
        lengths = [fmnist_length, len(fmnist_trainset) - fmnist_length], 
        generator=generator
    )

    joint_trainset = ConcatDataset([mnist_trainset_subsample, fmnist_trainset_subsample])
    joint_testset = ConcatDataset([mnist_testset, fmnist_testset])
    dataset_dict = {
        'subsample_1': mnist_trainset_subsample, 
        'subsample_2': fmnist_trainset_subsample,
        'joint_trainset' : joint_trainset, 
        'testsample_1' : mnist_testset,
        'testsample_2' : fmnist_testset,
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


def sampling_helper(root_dir, dataset):

    if(dataset == "MNIST"):
        return bias_mnist(root_dir=root_dir)
    else:
        return None


