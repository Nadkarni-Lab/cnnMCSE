"""File to run complexity based experiments. 
"""
from ntpath import join
import torch
from torch.utils.data import TensorDataset, random_split, ConcatDataset

from cnnMCSE.dataloader import dataloader_helper


def complexity_mnist(root_dir, 
    sample_sizes:list=[50000, 50000], 
    tl_transforms:bool=False,
    start_seed:int=42):

    generator = torch.Generator().manual_seed(start_seed)
    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    CIFAR10_trainset, CIFAR10_testset = dataloader_helper(dataset='CIFAR10', root_dir=root_dir, tl_transforms=tl_transforms)

    mnist_length, CIFAR10_length = sample_sizes

    mnist_trainset_subsample, _ = random_split(
        mnist_trainset, 
        lengths = [mnist_length, len(mnist_trainset) - mnist_length], 
        generator=generator
    )

    CIFAR10_trainset_subsample, _ = random_split(
        CIFAR10_trainset, 
        lengths = [CIFAR10_length, len(CIFAR10_trainset) - CIFAR10_length], 
        generator=generator
    )

    joint_trainset = ConcatDataset([mnist_trainset_subsample, CIFAR10_trainset_subsample])
    joint_testset = ConcatDataset([mnist_testset, CIFAR10_testset])
    dataset_dict = {
        'subsample_1': mnist_trainset_subsample, 
        'subsample_2': CIFAR10_trainset_subsample,
        'joint_trainset' : joint_trainset, 
        'testsample_1' : mnist_testset,
        'testsample_2' : CIFAR10_testset,
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


def complexity_helper(root_dir, dataset, tl_transforms: bool=False):

    if(dataset == "MNIST"):
        return complexity_mnist(root_dir=root_dir, tl_transforms=tl_transforms)
    else:
        return None