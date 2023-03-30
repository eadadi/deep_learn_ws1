from torch_cifar_loader import classes, get_trainloader_and_testloader, device
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part2_baseline import Test
from part2_optimization import Train, train_and_test

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def foward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

from part2_baseline import epochs, input_size, hidden_size, output_size, loss_fn, batch_size
from part2_optimization import lr, momentum
model = NeuralNetwork(input_size, hidden_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_and_test()
