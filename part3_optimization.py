import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part3_baseline import CNNetwork, Train, Test, epochs, loss_fn, batch_size, get_trainloader_and_testloader,device

def train_and_test(model,
        loss_fn,
        optimizer,
        epochs,
        batch_size,
        const_log = False):
    flag = False
    trainloader, testloader = get_trainloader_and_testloader(batch_size = batch_size)
    res_trainloss, res_traincorrect = list(), list()
    res_testloss, res_testcorrect = list(), list()

    for t in range(epochs):
        if t+1==epochs or const_log:
            print(f"[E{t+1}]", end=" ")
            flag = True 

        e_trainloss, e_traincorrect = Train(
                trainloader,
                model,
                loss_fn,
                optimizer)
        e_testloss, e_testcorrect = Test(
                testloader,
                model,
                loss_fn,
                flag)
        if flag:
            print(f"TrainLoss: {e_trainloss : >8f}")
        res_trainloss.append(round(e_trainloss,2))
        res_traincorrect.append(round(e_traincorrect,2))
        res_testloss.append(round(e_testloss,2))
        res_testcorrect.append(round(e_testcorrect,2))

    res = (res_trainloss, res_traincorrect, res_testloss, res_testcorrect)
    return res

#Use best params obtained in the previous question
momentum = .9
lr = .001
std = .1
def activate_optimization():
    model = CNNetwork(std)
    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(),lr),
            epochs=epochs,
            batch_size=batch_size)

#activate_optimization()
