from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim

import os
import sys
sys.path.append(os.curdir)

from utils import model_config,save_path
from svm.SVM_model import SVM
from svm.data_process import loader_data, prepare_for_analysis
import torch.nn as nn

device = model_config['device']
lr = model_config['lr']
c = model_config['c']
EPOCH = model_config['epoch']
dataFile = model_config['dataFile'] 


def save_model(model):
    torch.save(model.state_dict(), save_path)

def load_model(save_path):
    model = SVM().to(device)
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    return model

def loss(y,output):
    loss = 1-y * output
    loss[loss<=0] = 0
    return torch.sum(loss)

def train(train_loader):
    model = SVM().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()
    for epoch in range(EPOCH):
        sum_loss = 0

        for i,data in enumerate(train_loader,0):
            x = data[0].to(device)
            y = data[1].to(device)

            optimizer.zero_grad()
            output = model(x)

            
            # loss = loss(y,output)
            loss = torch.mean(torch.clamp(1 - output.t() * y, min=0))  # hinge loss
            loss += 0.01 * torch.mean(model.layer.weight ** 2)  # l2 penalty

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch:{:4d}\tloss: {}".format(epoch, sum_loss / len(train_set)))
    save_model(model)
    return model


def test(train_set,test_set):
    model = load_model(save_path)
    # train dataset
    x_train = train_set.tensor_data['X'].to(device)
    y_train = train_set.tensor_data['Y'].to(device).squeeze()

    y_pred_train = model.batch_pred(x_train)
    train_correct = (y_pred_train == y_train).sum().item()
    acc_train = train_correct / x_train.shape[0]

    # test dataset
    x_test = test_set.tensor_data['X'].to(device)
    y_test = test_set.tensor_data['Y'].to(device).squeeze()

    y_pred_train = model.batch_pred(x_test)
    acc_test = (y_pred_train == y_test).sum().item() / x_test.shape[0]

    print("Train Accuracy:\t{:.2%}".format(acc_train))
    print("Test Accuracy:\t{:.2%}".format(acc_test))
    for name,param in model.named_parameters():
        print(name, param)

def prediction():
    fp = open("./data/pre_data.csv", 'w')
    fp.write("ID,Percentage,Category\n")


    model = load_model(save_path)
    data = prepare_for_analysis(dataFile)
    y = data[:,:1].squeeze()
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:,1:]).astype('float32')
    X =torch.Tensor(X).to(device)

    S = torch.nn.Sigmoid()
    y_pred = S(model(X)).detach().numpy().squeeze()

    for sample in range(len(y_pred)):
        percent = y_pred[sample]
        predicted = 0
        if percent>.5:
            predicted = 1
        ground_truth = y[sample]
        model_correct = 1
        if predicted != ground_truth:
            model_correct = 0
        category = "NA";
        if (predicted, model_correct) == (0,0):
            category = "FN"
        elif (predicted, model_correct) == (0,1):
            category = "TN"
        elif (predicted, model_correct) == (1,0):
            category = "FP"
        elif (predicted, model_correct) == (1,1):
            category = "TP"
        
        fp.write(str(sample))
        fp.write(',')
        fp.write(str(percent))
        fp.write(',')
        fp.write(str(category))
        fp.write('\n')
    fp.close()



if __name__ == "__main__":
    train_loader,test_loader,train_set,test_set= loader_data()

    model = train(train_loader)
    print("=====================train model done.")
    test(train_set,test_set)

    # prediction()
   
