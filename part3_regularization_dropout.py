import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part3_optimization import CNNetwork, Test, Train, train_and_test
from part3_baseline import epochs, batch_size
from part3_optimization import lr, momentum, std

def activate_dropout():
    model = CNNetwork(std,dropout_flag=True)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

#activate_dropout()
