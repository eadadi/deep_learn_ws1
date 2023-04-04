import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part3_baseline import Test, CNNetwork, epochs,batch_size
from part3_optimization import Train, train_and_test, lr, momentum, std

def activate_width():
    widths = [(64,16),(256,64),(512,256)]
    results = []
    for hidden_size in widths:
        model = CNNetwork(std)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        r = train_and_test(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer=torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=momentum),
                epochs=epochs,
                batch_size=batch_size,
                const_log = True)
        results.append(r)
    return results

#activate_width()
