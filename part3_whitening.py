import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from part3_optimization import Test, Train, train_and_test
from part3_regularization_decay import CNNetwork
from sklearn.decomposition import PCA

class WhitenedNetwork(nn.Module):
    def __init__(self, model, whiten):
        super(WhitenedNetwork, self).__init__()
        self.model = model
        self.whiten_lambda_transformation = whiten

    def forward(self, x):
        whiten = self.whiten_lambda_transformation
        x = self.whiten(x)
        x = self.model(x)
        return x

from part3_baseline import epochs, loss_fn, batch_size
from part3_optimization import lr, momentum, std

def pca_batch(X):
    # Perform PCA
    pca = PCA(n_components=X.shape[1], whiten=True)
    pca.fit(X)
    eigenvectors = torch.from_numpy(pca.components_).float()

    # Define the whitening transform
    whiten = lambda x: torch.mm(x, eigenvectors.T)

def activate_whitening():
    model = CNNetwork(std)
    whitened_model = WhitenedNetwork(model, pca_batch)

    return train_and_test(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(
                whitened_model.parameters(),
                lr=lr,
                momentum=momentum),
            epochs=epochs,
            batch_size=batch_size)

#activate_whitening()
