"""Script to return all the computer vision datasets.
"""
import torch
import torchvision

from torchvision import transforms
from torch.utils.data import random_split

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

def synthetic_dataset(max_sample_size: int, n_informative: int, n_features: int, n_classes: int, train_test_split:float, seed:int=42):
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
    sample_dataset = make_classification(n_samples=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes)
    tensor_x = torch.Tensor(sample_dataset[0]) 
    tensor_y = torch.LongTensor(sample_dataset[1]) 
    dataset = TensorDataset(tensor_x,tensor_y) 
    trainset, testset = random_split(dataset, [train_test_split, 1 - train_test_split], generator=torch.Generator().manual_seed(42))

    return trainset, testset

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
    
    else:
        return None