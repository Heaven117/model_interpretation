import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs


from Data_Cleaning import prepare_for_analysis
device =  "mps" if torch.backends.mps.is_available() else "cpu" #change to cpu to reproduce cpu output 
DATABASE = 'data/'
DATASET = 'FICO'
batch_size = 5
EPOCH = 1
c = 0.01
lr = 0.1


def load_data(filename):
    data = prepare_for_analysis(filename)

    y = data[:,:1]
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:,1:])

    # -- Needs to be retained for inserting new samples
    mean = scaler.mean_
    scale = scaler.scale_

    num_samples , num_attributes = X.shape

    # -- Split Training/Test -- 
    X_train = X[:int(0.8*num_samples)]
    X_test = X[int(0.8*num_samples):]

    y_train = y[:int(0.8*num_samples)]
    y_test = y[int(0.8*num_samples):]

    return X_train,X_test,y_train,y_test,num_attributes

def save_model(model,saveName):
    torch.save(model.state_dict(), saveName)

def load_model(model,saveName):
    model.load_state_dict(torch.load(saveName))
    model.to(device)
    return model

def train(X_train,y_train, model,reTrain=False):
    saveName = DATABASE+f"svm_{DATASET}_{EPOCH}.pth"
    if(os.path.exists(saveName) and reTrain == False):
        load_model(model,saveName)
        return
       
    X = torch.FloatTensor(X_train)
    Y = torch.FloatTensor(y_train)
    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for epoch in range(EPOCH):
        sum_loss = 0

        for i in range(0, N, batch_size):
            x = X[i : i + batch_size].to(device)
            y = Y[i : i + batch_size].to(device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss += c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))
    save_model(model,saveName)
    
def visualize(X, Y, model):
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.0)] = 4
    z[np.where((z > 0.0) & (z <= 1.0))] = 3
    z[np.where((z > -1.0) & (z <= 0.0))] = 2
    z[np.where(z <= -1.0)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--c", type=float, default=0.01)
    # parser.add_argument("--lr", type=float, default=0.1)
    # parser.add_argument("--batch_size", type=int, default=5)
    # parser.add_argument("--epoch", type=int, default=10)
    # = parser.parse_)
    # print(

    fileName = DATABASE + 'FICO_final_data.csv'
    X_train,X_test,y_train,y_test ,num_attributes= load_data(fileName)

    model = nn.Linear(num_attributes, 1)
    model.to(device)
    
    train(X_train,y_train, model)
    visualize(X_train,y_train, model)

    # X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    # X = (X - X.mean()) / X.std()
    # Y[np.where(Y == 0)] = -1
    # train(X, Y, model)
    # visualize(X, Y, model)


