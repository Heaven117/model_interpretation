import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils import get_default_config
model_config = get_default_config()[0]
BATCH_SIZE = model_config['batch_size']

def prepare_for_analysis(filename):
	data_array = pd.read_csv(filename,header=None).values

	# -- Removes the columns with all -9 values -- 
	row_no = 0 
	for row in data_array:
		for col_i in range(1,row.shape[0]):
			if (row[col_i] == -9):
				remove = True
			else:
				remove = False
				break

		if remove:
			data_array = np.delete(data_array, row_no, 0)

		else:
			row_no += 1

	return data_array

class MyDataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __getitem__(self,index):
        return self.data['X'][index],self.data['Y'][index]
    def __len__(self):
        return len(self.data["X"])


filename = 'data/FICO_final_data.csv'
def loader_data():
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


def load_data():
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
    

    return X_train,X_test,y_train,y_test

