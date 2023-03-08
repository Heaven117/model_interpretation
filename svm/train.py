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

import sys
sys.path.append('./')
from IF.utils import *
from IF.IF_svm import *
from Data_Cleaning import prepare_for_analysis,MyDataset
from svm.SVM_model import SVM

device =  torch.device('cpu')
DATABASE = 'data/'
DATASET = 'FICO'
BATCH_SIZE = 5
EPOCH = 10
c = 0.01
lr = 0.1
saveName = DATABASE+f"svm_{DATASET}_{EPOCH}.pth"

def load_data():
    filename = DATABASE + 'FICO_final_data.csv'
    data = prepare_for_analysis(filename)

    y = data[:,:1]
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:,1:])
    # mean = scaler.mean_
    # scale = scaler.scale_

    num_samples = X.shape[0]

    # -- Split Training/Test -- 
    X_train = X[:int(0.8*num_samples)]
    X_test = X[int(0.8*num_samples):]

    y_train = y[:int(0.8*num_samples)]
    y_test = y[int(0.8*num_samples):]
    
    train_set = MyDataset({"X":torch.Tensor(X_train),"Y":torch.Tensor(y_train)})
    test_set = MyDataset({"X":torch.Tensor(X_test),"Y":torch.Tensor(y_test)})

    train_loader= DataLoader(train_set,batch_size=BATCH_SIZE)
    test_loader= DataLoader(test_set,batch_size=BATCH_SIZE)

    # return X_train,X_test,y_train,y_test,num_attributes
    return train_loader,test_loader

def save_model(model):
    torch.save(model.state_dict(), saveName)

def load_model():
    model = SVM().to(device)
    model.load_state_dict(torch.load(saveName))
    model.to(device)
    return model

def criterion(y,output,weight):
    loss = torch.mean(torch.clamp(1 - y * output, min=0))
    loss += c * (weight.t() @ weight) / 2.0
    return loss

    # loss = 1-y * output
    # loss[loss<=0] = 0
    # return torch.sum(loss)

def train(train_loader,test_loader):
    model = SVM().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(EPOCH):
        sum_loss = 0

        for i,data in enumerate(train_loader,0):
            x = data[0].to(device)
            y = data[1].to(device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.layer.weight.squeeze()

            # 折页损失
            loss = criterion(y,output,weight)
            # loss = torch.mean(torch.clamp(1 - y * output, min=0))
            # loss += c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / len(train_loader.dataset)))
    save_model(model)
    return model

def test(train_loader,test_loader,model):
    config = get_default_config()
    # init_logging('logfile.log')
    calc_main(config, model,train_loader,test_loader)

if __name__ == "__main__":
    train_loader,test_loader= load_data()

    if(os.path.exists(saveName)):
        model = load_model()
    else:
        model = train(train_loader,test_loader)

    test(train_loader,test_loader,model)
   
