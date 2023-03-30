import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part2_optimization import Test, Train, train_and_test, NeuralNetwork
from part2_baseline import epochs, input_size, hidden_size, output_size, batch_size
from part2_optimization import lr, momentum, std

class DropoutNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std):
        super(DropoutNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        init.normal_(self.fc1.weight, mean=0.0, std=std)
        init.normal_(self.fc2.weight, mean=0.0, std=std)

    def forward(self, x):
        dropout = nn.Dropout()
        x = self.fc1(x)
        x = dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def activate_dropout():
    model = DropoutNetwork(input_size, hidden_size, output_size, std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

#activate()
