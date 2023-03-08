import torch
import torch.optim as optim

import os
import sys
sys.path.append(os.curdir)

from utils import *
from IF.IF_svm import *
from svm.SVM_model import SVM
from svm.data_process import load_data

from utils import get_default_config
model_config = get_default_config()[0]
save_path = model_config['save_path']
device = model_config['device']
lr = model_config['lr']
c = model_config['c']
EPOCH = model_config['epoch']

def save_model(model):
    torch.save(model.state_dict(), save_path)

def load_model(save_path):
    model = SVM().to(device)
    model.load_state_dict(torch.load(save_path))
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

if __name__ == "__main__":
    train_loader,test_loader= load_data()

    model = train(train_loader,test_loader)

    print("train model done.")
   
