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

def ratio_mnist(root_dir, 
    ratios=[0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99],
    sample_sizes:list=[50000, 50000], 
    tl_transforms:bool=False,
    start_seed:int=42):

    generator = torch.Generator().manual_seed(start_seed)
    mnist_trainset, mnist_testset = dataloader_helper(dataset='MNIST', root_dir=root_dir, tl_transforms=tl_transforms)
    CIFAR10_trainset, CIFAR10_testset = dataloader_helper(dataset='CIFAR10', root_dir=root_dir, tl_transforms=tl_transforms)

    dataset_dict = {}
    balanced_testset = ConcatDataset([mnist_testset, CIFAR10_testset])
    dataset_dict['balanced_testset'] = balanced_testset
    dataset_dict['train_test_match'] = list()
    for ratio in ratios:
        subsample_name = f'joint_trainset_{str(ratio)}'
        mnist_length = int(ratio * sample_sizes[0])
        CIFAR10_length = int((1- ratio) * sample_sizes[1])
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
        joint_trainset_ratio = ConcatDataset([mnist_trainset_subsample, CIFAR10_trainset_subsample])
        dataset_dict[subsample_name] = joint_trainset_ratio
        dataset_dict['train_test_match'].append(
            [subsample_name,  'balanced_testset']
        )
    
    print(dataset_dict)


    # mnist_length, CIFAR10_length = sample_sizes

    # mnist_trainset_subsample, _ = random_split(
    #     mnist_trainset, 
    #     lengths = [mnist_length, len(mnist_trainset) - mnist_length], 
    #     generator=generator
    # )

    # CIFAR10_trainset_subsample, _ = random_split(
    #     CIFAR10_trainset, 
    #     lengths = [CIFAR10_length, len(CIFAR10_trainset) - CIFAR10_length], 
    #     generator=generator
    # )

    # joint_trainset = ConcatDataset([mnist_trainset_subsample, CIFAR10_trainset_subsample])
    
    # dataset_dict = {
    #     'subsample_1': mnist_trainset_subsample, 
    #     'subsample_2': CIFAR10_trainset_subsample,
    #     'joint_trainset' : joint_trainset, 
    #     'testsample_1' : mnist_testset,
    #     'testsample_2' : CIFAR10_testset,
    #     'joint_testset' : joint_testset,
    #     'train_test_match': [
    #         ['subsample_1', 'testsample_1'], 
    #         ['subsample_2', 'testsample_2'], 
    #         ['joint_trainset', 'joint_testset'], 
    #         ['joint_trainset', 'testsample_1'],
    #         ['joint_trainset', 'testsample_2']
    #     ]
    # }

    return dataset_dict


def sampling_helper(root_dir, dataset, tl_transforms:bool=False):

    if(dataset == "MNIST"):
        return bias_mnist(root_dir=root_dir, tl_transforms=tl_transforms)
    if(dataset == "MNIST_ratios"):
        return ratio_mnist(root_dir=root_dir, tl_transforms=tl_transforms)
    else:
        return None


