import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import csv

import sys
import os
sys.path.append(os.curdir)

from utils.parser import *
from utils.helper import display_progress
from models.data_process import *

args = parse_args()
device = args.device

class MLP(nn.Module) :
    def __init__(self) :
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(102, 64), 
                                nn.ReLU(), 
                                nn.Linear(64, 32), 
                                nn.ReLU(),
                                nn.Linear(32, 2)
                                )
    def forward(self, x) :
        out = self.net(x) 
        return F.softmax(out)
    
    def predict_single(self,x):
        out = self.forward(x)
        _, pred = torch.max(out,dim = 0)
        return pred.item()
    
    def predict_anchor(self,x,encoder):
        x = encoder_process(x,encoder)
        x = normalize(x,axis = 0,norm = 'max')
        x = torch.from_numpy(x)
        out = self.forward(x)
        _, pred = torch.max(out,dim = 1)
        return pred.numpy()


def train_MLP(train_loader,test_loader):
    # writer = SummaryWriter(log_dir = args.log_dir)
    os.makedirs(args.model_path,exist_ok=True)


    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = 0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epoch) :
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        test_id_num = 0
        for x, label in train_loader :
            x, label = x.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(out, 1)
            # print(pred)
            num_correct = (pred == label).sum().item()
            acc = num_correct / x.shape[0]
            train_acc += acc

            display_progress(f"train. MLP model : ", test_id_num, len(train_loader))
            test_id_num +=1

            
        print(f'epoch : {epoch + 1}, train_loss : {train_loss / len(train_loader.dataset)}, train_acc : {train_acc / len(train_loader)}')
        # writer.add_scalar('train_loss', train_loss / len(train_loader.dataset), epoch)

        # 验证
        # mse_loss = 1000000
        # with torch.no_grad() :
        #     total_loss = []
        #     model.eval()
        #     for x, label in test_loader :
        #         x, label = x.to(device), label.to(device)
        #         out = model(x)
        #         loss = criterion(out, label)
        #         total_loss.append(loss.item())
            
        #     val_loss = sum(total_loss) / len(total_loss)
        
        # if val_loss < mse_loss :
        #     mse_loss = val_loss 
        #     torch.save(model.state_dict(), args.model_path+f'MPL_{args.epoch}.pth')
    torch.save(model.state_dict(), args.model_path+f'MPL_{args.epoch}.pth')

def test_MLP(test_loader):
    # writer = SummaryWriter(log_dir = args.log_dir)
    os.makedirs(args.predict_path,exist_ok=True)

    best_model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    ckpt = torch.load(args.model_path+f'MPL_{args.epoch}.pth', map_location='cpu')
    best_model.load_state_dict(ckpt)

    test_loss = 0.0
    test_acc = 0.0
    result = []
    best_model.eval()
    for x, label in test_loader :
        x, label = x.to(device), label.to(device)
        out = best_model(x)
        loss = criterion(out, label)

        test_loss+= loss.item()
        _, pred = torch.max(out, 0)
        result.append([out[0].item(),(pred == label).item(),pred.item()])
        num_correct = (pred == label).sum().item()
        acc = num_correct / x.shape[0]
        test_acc += acc
    
    print(f'test_loss : {test_loss / len(test_loader.dataset)}, test_acc : {test_acc / len(test_loader)}')
    # result = torch.cat(result, dim = 0).cpu().numpy()

    df = pd.DataFrame(result,columns=['percentage','category','prediction'])
    for i, res in enumerate(result) :
        out,model_correct,predicted= res
        category = "NA"
        if (predicted, model_correct) == (0,0):
            category = "FN"
        elif (predicted, model_correct) == (0,1):
            category = "TN"
        elif (predicted, model_correct) == (1,0):
            category = "FP"
        elif (predicted, model_correct) == (1,1):
            category = "TP"
        df.iloc[i,1] = category

    df.to_csv(args.predict_path+f'pred_data000.csv',index='id',float_format='%.6f',index_label='id')
    
            

def load_model(baseDir=None):
    best_model = MLP().to(device)
    ckpt = torch.load(args.model_path+f'MPL_{args.epoch}.pth', map_location='cpu')
    best_model.load_state_dict(ckpt)
    return best_model


def predictAllData():
    dataset_loader = Adult_data(mode = 'none')
    test_MLP(dataset_loader)

if __name__ == "__main__":
    train_dataset = Adult_data(mode = 'train')
    test_dataset = Adult_data(mode = 'test')

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last = False)
    # train_MLP(train_loader,test_loader)


    # test_MLP(test_loader)
    predictAllData()


    
    


