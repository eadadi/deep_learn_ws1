import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part3_optimization import CNNetwork,Test, Train, train_and_test, lr, momentum,std
from part3_baseline import epochs, loss_fn, batch_size
 
def activate_xavier():
    model = CNNetwork(std=std,
            iinit=torch.nn.init.xavier_normal_,iinit_flag=False)
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

#activate_xavier()
