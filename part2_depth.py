from torch_cifar_loader import classes, get_trainloader_and_testloader, device
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part2_optimization import train_and_test

from part2_baseline import epochs, input_size, hidden_size, output_size, loss_fn, batch_size
from part2_optimization import lr, momentum, std

class Neural3Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std):
        super(Neural3Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        init.normal_(self.fc1.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def activate3():
    model = Neural3Network(input_size, hidden_size, output_size, std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

class Neural4Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std):
        super(Neural4Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        init.normal_(self.fc1.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

def activate4():
    model = Neural4Network(input_size, hidden_size, output_size, std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

class Neural10Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std):
        super(Neural10Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, hidden_size)
        self.fc8 = nn.Linear(hidden_size, hidden_size)
        self.fc9 = nn.Linear(hidden_size, hidden_size)
        self.fc10 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        init.normal_(self.fc1.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.fc10(x)
        return x

def activate10():
    model = Neural10Network(input_size, hidden_size, output_size, std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

