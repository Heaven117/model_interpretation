import pandas as pd
import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from server.utils import get_default_config
model_config = get_default_config()[0]
dataFile = model_config['dataFile'] 

class MyDataset(Dataset):
    def __init__(self,data):
        self.raw_data=data
        # 标准化
        scaler = StandardScaler()
        x_scale = scaler.fit_transform(data['X']).astype('float32')
        self.mean = scaler.mean_
        self.scale = scaler.scale_

        self.tensor_data=copy.deepcopy(data)
        self.tensor_data['X'] = torch.Tensor(x_scale)
        self.tensor_data['Y'] = torch.Tensor(data['Y'])

        self.scale_data=copy.deepcopy(data)
        self.scale_data['X'] = x_scale

        

    def __getitem__(self,index):
        return self.tensor_data['X'][index],self.tensor_data['Y'][index]
    
    def getRawData(self,index = None):
        if(index is None):
            return self.raw_data['X'],self.raw_data['Y']
        return self.raw_data['X'][index],self.raw_data['Y'][index]
    
    def getScaleData(self,index = None):
        if(index is None):
              return self.scale_data['X'],self.scale_data['Y']
        return self.scale_data['X'][index],self.scale_data['Y'][index]
    
    def __len__(self):
        return len(self.data["X"])
    
    def scaled_row(self,row):
        scld = []
        for k in range(row.shape[0]):
            scld.append((row[k] - self.mean[k])/self.scale[k])
        scld = np.array(scld)
        
        return np.array(scld)



def loader_data(batch_size = 1,filename = dataFile):
    data = prepare_for_analysis(filename)

    y = data[:,:1]
    X = data[:,1:]
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    num_samples = X.shape[0]

    # -- Split Training/Test -- 
    X_train = X[:int(0.8*num_samples)]
    X_test = X[int(0.8*num_samples):]

    y_train = y[:int(0.8*num_samples)]
    y_train[np.where(y_train == 0)] = -1
    y_test = y[int(0.8*num_samples):]
    y_test[np.where(y_test == 0)] = -1

    
    # train_set = MyDataset({"X":torch.Tensor(X_train),"Y":torch.Tensor(y_train)})
    # test_set = MyDataset({"X":torch.Tensor(X_test),"Y":torch.Tensor(y_test)})
    train_set = MyDataset({"X":X_train,"Y":y_train})
    test_set = MyDataset({"X":X_test,"Y":y_test})


    train_loader= DataLoader(train_set,batch_size=batch_size)
    test_loader= DataLoader(test_set,batch_size=batch_size)

    # return X_train,X_test,y_train,y_test,num_attributes
    return train_loader,test_loader,train_set,test_set


def load_data(filename = dataFile):
    data = prepare_for_analysis(filename)

    y = data[:,:1]
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:,1:]).astype('float32')
    # mean = scaler.mean_
    # scale = scaler.scale_

    num_samples = X.shape[0]

    # -- Split Training/Test -- 
    X_train = X[:int(0.8*num_samples)]
    X_test = X[int(0.8*num_samples):]

    y_train = y[:int(0.8*num_samples)]
    y_test = y[int(0.8*num_samples):]
    

    return X_train,X_test,y_train,y_test



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
