import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part2_optimization import Test, Train, train_and_test, lr, momentum
from part2_baseline import epochs, input_size, hidden_size, output_size, loss_fn, batch_size

class XavierNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(XavierNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with Xavier initialization
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def activate_xavier():
    model = XavierNetwork(input_size, hidden_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

