"""Script to return all the computer vision datasets.
"""
import torchvision

def mnist_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.MNIST(root=root_dir,
        train=True,
        download=True,
        transform=transform)

    testset = torchvision.datasets.MNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)
    
    return trainset, testset

def fmnist_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FashionMNIST(root = root_dir,
        train=True,
        download=True,
        transform=transform)

    testset = torchvision.datasets.FashionMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def kmnist_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.KMNIST(root=root_dir,
        train=True,
        download=True,
        transform=transform)

    testset  = torchvision.datasets.KMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def emnist_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.EMNIST(root=root_dir,
        train=True,
        download=True,
        transform=transform)

    testset = torchvision.datasets.EMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def qmnist_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.QMNIST(root=root_dir,
        train=True,
        download=True,
        transform=transform)

    testset = torchvision.datasets.QMNIST(root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def cifar10_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(
        root = root_dir,
        train=True,
        download=True,
        transform=transform)


    testset = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=True,
        transform=transform)


    return trainset, testset

def stl10_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.STL10(
        root = root_dir,
        train=True,
        download=True,
        transform=transform)

    testset = torchvision.datasets.STL10(
        root=root_dir,
        train=False,
        download=True,
        transform=transform)

    return trainset, testset

def fake_dataset(root_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FakeData(
        root = root_dir,
        train=True,
        download=True,
        transform=transform,
        size=60000,
        image_size=(1, 28, 28))

    testset = torchvision.datasets.FakeData(
        root=root_dir,
        train=False,
        download=True,
        transform=transform,
        size=10000,
        image_size=(1, 28, 28))

    return trainset, testset


def dataloader_helper(dataset, root_dir):
    if(dataset == "MNIST"):
        return mnist_dataset(root_dir)
    
    if(dataset == "FMNIST"):
        return fmnist_dataset(root_dir)

    if(dataset == "KMNIST"):
        return kmnist_dataset(root_dir)
    
    if(dataset == "EMNIST"):
        return emnist_dataset(root_dir)
    
    if(dataset == "QMNIST"):
        return qmnist_dataset(root_dir)
    
    if(dataset == "CIFAR10"):
        return cifar10_dataset(root_dir)
    
    if(dataset == "STL10"):
        return stl10_dataset(root_dir)
    
    if(dataset == "FAKE"):
        return fake_dataset(root_dir)


    pass
