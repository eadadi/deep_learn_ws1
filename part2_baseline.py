from torch_cifar_loader import classes, get_trainloader_and_testloader, device
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, std):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        init.normal_(self.fc1.weight, mean=0.0, std=std)
        init.normal_(self.fc2.weight, mean=0.0, std=std)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def Train(dataloader, model, loss_fn, input_size, optimizer):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    train_loss, train_correct = 0, 0

    for batch, (X,y) in enumerate(dataloader):
        #Normalize X:
        X = X.reshape(len(X), input_size)
        X = (X-X.min())/(X.max()-X.min())
        X,y = X.to(device), y.to(device)

        #Compute prediciton error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= num_batches
    train_correct /= size
    return train_loss, train_correct

def Test(dataloader, model, loss_fn, log=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #Normalize X:
            X = X.reshape(len(X), input_size) 
            X = (X-X.min())/(X.max()-X.min())
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if log:
        print(f"Accuracy: {(100*correct): >0.1f}%,\t AvgLoss: {test_loss: >8f},\t ", end ="")
    return test_loss, correct


epochs = 60
input_size = 32 * 32 * 3
hidden_size = 256
output_size = len(classes)
loss_fn = nn.CrossEntropyLoss()
batch_size = 64
params = {
        "std": [.01],
        "lr": [.001, .0001],
        "acc":[.1, .01]
        }

def get_grid_search_triplets(params):
    result = []
    l1 = params["acc"]
    l2 = params["lr"]
    l3 = params["std"]
    for v1 in l1:
        for v2 in l2:
            for v3 in l3:
                result.append((v1,v2,v3))
    return result

def train_and_test_for_params(params):
    flag = False
    trainloader, testloader = get_trainloader_and_testloader(batch_size = batch_size)
    grid_search_params = get_grid_search_triplets(params)
    res = results_by_params_over_epoch = {} 
    for params_set in grid_search_params:
        res[params_set] = (list(),list(),list(),list())
        acc,lr,std = params_set
        print(f"(acc,lr,std)={params_set}".center(20,'='))

        """
        Set the model by this std parameter
        """
        model = NeuralNetwork(input_size, hidden_size, output_size, std)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=acc)
        for t in range(epochs):
            if t+1==epochs:
                print(f"[E{t+1}]", end=" ")
                flag = True
            e_trainloss, e_traincorrect = Train(trainloader,
                    model,
                    loss_fn,
                    input_size,
                    optimizer)
            e_testloss, e_testcorrect = Test(testloader,
                    model,
                    loss_fn,
                    flag)
            if flag:
                print(f"TrainLoss: {e_trainloss: >8f}")
            res[params_set][0].append(round(e_trainloss,2))
            res[params_set][1].append(round(e_traincorrect,2))
            res[params_set][2].append(round(e_testloss,2))
            res[params_set][3].append(round(e_testcorrect,2))
            flag = False

    return res
            
#train_and_test_for_params(params)
#.print("Done!")
