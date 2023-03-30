import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part2_baseline import Test, NeuralNetwork, epochs, input_size, output_size, batch_size
from part2_optimization import Train, train_and_test, lr, momentum, std

def activate_width():
    widths = [64, 1024, 4096]
    results = []
    for hidden_size in widths:
        model = NeuralNetwork(input_size, hidden_size, output_size, std)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        r = train_and_test(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=momentum),
                epochs=epochs,
                batch_size=batch_size)
        results.append(r)
    return results

#activate()
