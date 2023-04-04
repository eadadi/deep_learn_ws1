import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part3_optimization import train_and_test

from part3_baseline import epochs, loss_fn, batch_size, CNNetwork
from part3_optimization import lr, momentum, std

class CNN3etwork(nn.Module):
    def __init__(self,std,iinit=init.normal_,iinit_flag=True, dropout_flag=False):
        super(CNN3etwork, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=784)
        self.fc2 = nn.Linear(in_features=784, out_features=10)

        # Define the activation function
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        if iinit_flag:
            iinit(self.fc1.weight, mean=0.0, std=std)
            iinit(self.fc2.weight, mean=0.0, std=std)
            iinit(self.conv1.weight, mean=0.0, std=std)
            iinit(self.conv2.weight, mean=0.0, std=std)
            iinit(self.conv3.weight, mean=0.0, std=std)
        else:
            iinit(self.fc1.weight)
            iinit(self.fc2.weight)
            iinit(self.conv1.weight)
            iinit(self.conv2.weight)

        self.dropout_flag=dropout_flag

    def forward(self, x):
        # Pass the input through the convolutional layers and activation function
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 16 * 8 * 8)

        # Pass the flattened output through the fully connected layers and activation function
        x = self.relu(self.fc1(x))
        if self.dropout_flag:
            x = dropout(x)
        x = self.fc2(x)

        # Return the output
        return x

class CNN4etwork(nn.Module):
    def __init__(self,std,iinit=init.normal_,iinit_flag=True, dropout_flag=False):
        super(CNN4etwork, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Define the pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=784)
        self.fc2 = nn.Linear(in_features=784, out_features=10)

        # Define the activation function
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        if iinit_flag:
            iinit(self.fc1.weight, mean=0.0, std=std)
            iinit(self.fc2.weight, mean=0.0, std=std)
            iinit(self.conv1.weight, mean=0.0, std=std)
            iinit(self.conv2.weight, mean=0.0, std=std)
            iinit(self.conv3.weight, mean=0.0, std=std)
            iinit(self.conv4.weight, mean=0.0, std=std)
        else:
            iinit(self.fc1.weight)
            iinit(self.fc2.weight)
            iinit(self.conv1.weight)
            iinit(self.conv2.weight)

        self.dropout_flag=dropout_flag

    def forward(self, x):
        # Pass the input through the convolutional layers and activation function
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 16 * 8 * 8)

        # Pass the flattened output through the fully connected layers and activation function
        x = self.relu(self.fc1(x))
        if self.dropout_flag:
            x = dropout(x)
        x = self.fc2(x)

        # Return the output
        return x

class CNN5etwork(nn.Module):
    def __init__(self,std,iinit=init.normal_,iinit_flag=True, dropout_flag=False):
        super(CNN5etwork, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1)

        # Define the pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=784)
        self.fc2 = nn.Linear(in_features=784, out_features=10)

        # Define the activation function
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        if iinit_flag:
            iinit(self.fc1.weight, mean=0.0, std=std)
            iinit(self.fc2.weight, mean=0.0, std=std)
            iinit(self.conv1.weight, mean=0.0, std=std)
            iinit(self.conv2.weight, mean=0.0, std=std)
            iinit(self.conv3.weight, mean=0.0, std=std)
            iinit(self.conv4.weight, mean=0.0, std=std)
            iinit(self.conv5.weight, mean=0.0, std=std)
        else:
            iinit(self.fc1.weight)
            iinit(self.fc2.weight)
            iinit(self.conv1.weight)
            iinit(self.conv2.weight)

        self.dropout_flag=dropout_flag

    def forward(self, x):
        # Pass the input through the convolutional layers and activation function
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        x = self.relu(self.conv5(x))
        x = self.pool5(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 16 * 8 * 8)

        # Pass the flattened output through the fully connected layers and activation function
        x = self.relu(self.fc1(x))
        if self.dropout_flag:
            x = dropout(x)
        x = self.fc2(x)

        # Return the output
        return x

def activate3():
    model = CNN3etwork(std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

def activate4():
    model = CNN4etwork(std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

def activate5(const_log=False):
    model = CNN5etwork(std=std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size,
            const_log=const_log)

