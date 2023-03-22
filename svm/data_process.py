import pandas as pd
import numpy as np
import os
import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.helper import model_config
dataFile = model_config['dataFile'] 
batch_size = model_config['batch_size']

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
        return len(self.scale_data["X"])
    
    def scaled_row(self,row):
        scld = []
        for k in range(row.shape[0]):
            scld.append((row[k] - self.mean[k])/self.scale[k])
        scld = np.array(scld)
        
        return np.array(scld)



def loader_data(batch_size = batch_size,filename = dataFile):
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


def load_adult_income_dataset(filename = dataFile):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    if(os.path.exists(filename)):
         raw_data = np.genfromtxt(filename,delimiter=', ', dtype=str, invalid_raise=False)
    else:
         raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                             delimiter=', ', dtype=str, invalid_raise=False)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    # adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    return adult_data

def loader_adult_income_dataset():
    dataset = load_adult_income_dataset()
    target = dataset["income"]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=0, stratify=target)
    x_train = train_dataset.drop('income', axis=1)
    x_test = test_dataset.drop('income', axis=1)

    train_set = MyDataset({"X":x_train,"Y":y_train})
    test_set = MyDataset({"X":x_test,"Y":y_test})

    train_loader= DataLoader(train_set,batch_size=batch_size)
    test_loader= DataLoader(test_set,batch_size=batch_size)

    return train_loader,test_loader,train_set,test_set

