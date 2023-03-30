import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part2_optimization import Test, Train, train_and_test, NeuralNetwork
from part2_baseline import epochs, input_size, hidden_size, output_size, loss_fn, batch_size
from part2_optimization import lr, momentum, std

def activate_decay():
    model = NeuralNetwork(input_size, hidden_size, output_size, std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=1e-5),
            epochs=epochs,
            batch_size=batch_size)

#activate()
