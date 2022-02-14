"""Script to return all the computer vision datasets.
"""

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
    

def dataloader_helper(dataset, root_dir):
    if(dataset == "MNIST"):
        return mnist_dataset(root_dir)

    pass
