import torch
import torchvision
import numpy as np

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using {device} device")


"""
Load the datasets, return trainloader and testloader
"""
def get_trainloader_and_testloader(batch_size = 64):
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    trainset = datasets.CIFAR10(
        root="torch_cifar10",
        train=True,
        download=True,
        transform=ToTensor()
        )

    idxes = list(np.random.randint(len(trainset), size = 5000))
    trainset = torch.utils.data.Subset(trainset,idxes) 

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root="torch_cifar10",
        train=False,
        download=True,
        transform=ToTensor()
        )

    idxes = list(np.random.randint(len(testset), size = 1000))
    testset = torch.utils.data.Subset(testset,idxes) 

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
            shuffle=True, num_workers=2)

    return trainloader, testloader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
