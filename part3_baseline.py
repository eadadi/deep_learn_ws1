from torch_cifar_loader import classes, get_trainloader_and_testloader, device
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CNNetwork(nn.Module):
    def __init__(self,std,iinit=init.normal_,iinit_flag=True, dropout_flag=False, width_conv1=64, width_conv2=16):
        super(CNNetwork, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width_conv1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=width_conv1, out_channels=width_conv2, kernel_size=3, stride=1, padding=1)

        # Define the pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=width_conv2 * 8 * 8, out_features=784)
        self.fc2 = nn.Linear(in_features=784, out_features=10)

        # Define the activation function
        self.relu = nn.ReLU()

        # Initialize the weights with a zero-mean Gaussian distribution
        if iinit_flag:
            iinit(self.fc1.weight, mean=0.0, std=std)
            iinit(self.fc2.weight, mean=0.0, std=std)
            iinit(self.conv1.weight, mean=0.0, std=std)
            iinit(self.conv2.weight, mean=0.0, std=std)
        else:
            iinit(self.fc1.weight)
            iinit(self.fc2.weight)
            iinit(self.conv1.weight)
            iinit(self.conv2.weight)

        self.dropout_flag=dropout_flag

    def forward(self, x):
        # Pass the input through the convolutional layers and activation function
        dropout = nn.Dropout()
        x = self.relu(self.conv1(x))
        if self.dropout_flag:
            x = dropout(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        if self.dropout_flag:
            x = dropout(x)
        x = self.pool2(x)

        # Flatten the output of the convolutional layers
        x = x.view(-1, 16 * 8 * 8)

        # Pass the flattened output through the fully connected layers and activation function
        x = self.relu(self.fc1(x))
        if self.dropout_flag:
            x = dropout(x)
        x = self.fc2(x)

        # Return the output
        return x

def Train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    train_loss, train_correct = 0, 0

    for batch, (X,y) in enumerate(dataloader):
        #Normalize X:
        #X = X.reshape(len(X), )
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
            #X = X.reshape(len(X), ) 
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


epochs = 50
loss_fn = nn.CrossEntropyLoss()
batch_size = 64
params = {
        "std": [.1,.9],
        "lr": [.001,.01,.1],
        "acc":[.5,.9,.99]
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

def train_and_test_for_params(params, const_log=False):
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
        model = CNNetwork(std)
        for t in range(epochs):
            if t+1==epochs or const_log:
                print(f"[E{t+1}]", end=" ")
                flag = True
            e_trainloss, e_traincorrect = Train(trainloader,
                    model,
                    loss_fn,
                    optimizer=torch.optim.SGD(model.parameters(), lr=lr, momentum=acc))
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
