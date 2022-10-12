"""Script to return all the computer vision datasets.
"""
import torch
import torchvision
import numpy as np

from torchvision import transforms
from torch.utils.data import random_split, WeightedRandomSampler, TensorDataset
from sklearn.datasets import make_classification

def mnist_dataset(root_dir:str, tl_transforms:bool=False):
    """MNIST dataset. 

    Args:
        root_dir (str): Path to the root directory of the dataset. 
        tl_transforms (bool, optional): Transfer learning transforms. Defaults to False.

    Returns:
        torchvision.datasets, torchvision.datasets: A torchvision dataset appropriately transformed.  
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.MNIST(root=root_dir,
        train=True,
        download=False,
        transform=transform)

    testset = torchvision.datasets.MNIST(root=root_dir,
        train=False,
        download=False,
        transform=transform)
    
    return trainset, testset

def fmnist_dataset(root_dir:str, tl_transforms:bool=False):
    transform = transforms.Compose([transforms.ToTensor()])
    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.FashionMNIST(root = root_dir,
        train=True,
        download=False,
        transform=transform)

    testset = torchvision.datasets.FashionMNIST(root=root_dir,
        train=False,
        download=False,
        transform=transform)

    return trainset, testset

def kmnist_dataset(root_dir:str, tl_transforms:bool=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.KMNIST(root=root_dir, train=True, download=True, transform=transform)

    testset  = torchvision.datasets.KMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def emnist_dataset(root_dir:str, tl_transforms:bool=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.EMNIST(root=root_dir,
        train=True,
        download=True,
        transform=transform,
        split='mnist')

    testset = torchvision.datasets.EMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform,
        split='mnist')

    return trainset, testset

def qmnist_dataset(root_dir:str, tl_transforms:bool=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    trainset = torchvision.datasets.QMNIST(root=root_dir,
        train=True,
        download=True,
        transform=transform)

    testset = torchvision.datasets.QMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def cifar10_dataset(root_dir:str, tl_transforms:bool=False):
    transform = transforms.Compose([transforms.ToTensor()])

    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])
    else:
        to_grayscale = transforms.Grayscale()
        resize = transforms.Resize((28,28))
        normalize = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([
            resize,
            to_grayscale,
            transforms.ToTensor(),
            normalize
        ])

    trainset = torchvision.datasets.CIFAR10(root = root_dir,train=True,download=False,transform=transform)


    testset = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=False,
        transform=transform)


    return trainset, testset

def stl10_dataset(root_dir:str, tl_transforms:bool=False):    

    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])
    else:
        to_grayscale = transforms.Grayscale()
        resize = transforms.Resize((28,28))
        normalize = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([
            resize,
            to_grayscale,
            transforms.ToTensor(),
            normalize
        ])

    trainset = torchvision.datasets.STL10(
        root = root_dir,
        split='train',
        download=True,
        transform=transform)

    testset = torchvision.datasets.STL10(
        root=root_dir,
        split='test',
        download=True,
        transform=transform)

    return trainset, testset

def fake_dataset(root_dir:str, tl_transforms:bool=False):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if(tl_transforms):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    dataset = torchvision.datasets.FakeData(
        transform=transform,
        size=70000,
        image_size=(1, 28, 28),
        num_classes=10
    )
    
    trainset, testset = random_split(dataset, [60000, 10000])

    return trainset, testset

def synthetic_dataset(max_sample_size: int, n_informative: int, n_features: int, n_classes: int, train_test_split:int, flip_y:float=0.1, seed:int=42):
    """Generate a synthetic dataset. 

    Args:
        max_sample_size (int): Maximum sample size of the dataset. 
        n_informative (int): Number of informative features. 
        n_features (int): Number of total features. 
        n_classes (int): Number of total classes. 
        train_test_split (float): Ratio of training to test. 
        seed (int, optional): Seed of the dataset. Defaults to 42.

    Returns:
        trainset, testset: Training and testing dataset. 
    """
    print('max_sample_size', max_sample_size)
    print('n_informative', n_informative)
    print('n_features', n_features)
    print('n_classes', n_classes)

    sample_dataset = make_classification(n_samples=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, flip_y=flip_y)
    tensor_x = torch.Tensor(sample_dataset[0]) 
    tensor_y = torch.LongTensor(sample_dataset[1]) 
    dataset = TensorDataset(tensor_x,tensor_y) 
    trainset, testset = random_split(dataset, [train_test_split, max_sample_size - train_test_split], generator=torch.Generator().manual_seed(42))

    return trainset, testset

def get_labels(subset):
    labels = list()
    for index, (X, y) in enumerate(subset):
        labels.append(y)
    
    return labels


def weighted_sampler(subset, mode:str):
    label_map = {
        0 : ["None", 1], 
        1 : ["Pneumonia", 1],
        2 : ["Pneumothorax", 2],
        3 : ["Atelectasis", 0.5],
        4 : ["Cardiomegaly", 0.5],
        5 : ["Consolidation", 2],
        6 : ["Edema", 1],
        7 : ["Enlarged CM", 2],
        8 : ["Opacity", 0.5],
        9 : ["Effusion", 0.5]
    }

    if(mode == "frequency"):
        #y_train_indices = subset.indices
        #all_dataset = subset.dataset

        y_train = get_labels(subset)
        print(y_train)
        unique_labels = np.unique(y_train)
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in unique_labels])
        print(class_sample_count)
        weights = 1. / class_sample_count
        weight_dict = {unique_label : weight for unique_label, weight in zip(unique_labels, weights)}
        print(weight_dict)
        samples_weight = np.array([weight_dict[t] for t in y_train])
        print(samples_weight)
        samples_weight = torch.from_numpy(samples_weight)

    if(mode == "mcse"):
        #y_train_indices = subset.indices
        y_train = get_labels(subset)
        samples_weight = np.array([label_map[t][1] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
    
    if(mode == "joint"):
        y_train = get_labels(subset)
        unique_labels = np.unique(y_train)
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in unique_labels])
        print(class_sample_count)
        weights = 1. / class_sample_count
        weight_dict = {unique_label : weight for unique_label, weight in zip(unique_labels, weights)}
        samples_weight = np.array([label_map[t][1] * weight_dict[t]  for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)


    return WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))





def dataloader_helper(dataset, root_dir, tl_transforms:bool=False):
    if(dataset == "MNIST"):
        return mnist_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif(dataset == "FMNIST"):
        return fmnist_dataset(root_dir, tl_transforms=tl_transforms)

    elif(dataset == "KMNIST"):
        return kmnist_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif(dataset == "EMNIST"):
        return emnist_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif(dataset == "QMNIST"):
        return qmnist_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif(dataset == "CIFAR10"):
        return cifar10_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif(dataset == "STL10"):
        return stl10_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif(dataset == "FAKE"):
        return fake_dataset(root_dir, tl_transforms=tl_transforms)
    
    elif("synthetic" in dataset):
        args = dataset.split(",")
        max_sample_size = int(args[1])
        n_informative = int(args[2])
        n_features = int(args[3]) 
        n_classes = int(args[4])
        train_test_split = 5000 / max_sample_size

        return synthetic_dataset
    
    else:
        return None