import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn


from sklearn.preprocessing import StandardScaler
# from sklearn import svm
# import joblib

device = torch.device('cpu')


# 定义sign激活函数进行预测，注意：计算损失时，不经过激活层
def pred(x):
	x[x>=0] = 1
	x[x<0] = -1
	return x

# 计算损失
def loss_func(scores, label):
	loss = 1-label*scores
	loss[loss<=0] = 0
	return torch.sum(loss)

class ModelError(Exception):
    pass

class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.layer = nn.Linear(23,1)

    def forward(self, x):
        x = self.layer(x)
        return x
    
class Model_Wrap():
	def __init__ (self, data, file_name = None):
		if (file_name != None):
			data = np.load(file_name)['arr_0']

		self.model = SVM().to(device)

		# -- Seperate --
		self.y = data[:,:1]
		scaler = StandardScaler()
		self.X = scaler.fit_transform(data[:,1:])

		# -- Needs to be retained for inserting new samples
		self.mean = scaler.mean_
		self.scale = scaler.scale_

		self.num_samples , self.num_attributes = self.X.shape

		# -- Split Training/Test -- 
		self.X_train = self.X[:int(0.8*self.num_samples)]
		self.X_test = self.X[int(0.8*self.num_samples):]

		self.y_train = self.y[:int(0.8*self.num_samples)]
		self.y_test = self.y[int(0.8*self.num_samples):]
	

	def train(self):
		optimizer = optim.SGD(self.model.parameters(), lr=0.01)

		for epoch in range(10000):  # loop over the dataset multiple times			
			inputs, targets = self.X_train, self.y_train
			inputs = Variable(torch.from_numpy(inputs).float(), requires_grad=False)
			label = Variable(torch.from_numpy(targets).int(), requires_grad=False)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = self.model(inputs).squeeze(1)
			loss = loss_func(outputs, label)
			loss.backward()
			optimizer.step()

			# print statistics
			if epoch % 1000 == 0:    # print every 2000 mini-batches
				# **todo 计算分类的准确率
				inputs, targets = self.X_test, self.y_test
				inputs = Variable(torch.from_numpy(inputs).float(), requires_grad=False)
				label = Variable(torch.from_numpy(targets).int(), requires_grad=False)
				scores = self.model(inputs).squeeze(1)
				num_correct = (pred(scores) == label).sum()
				acc = num_correct*100.0 / inputs.shape[0]
				print("loss=",loss.detach().numpy(),"acc=", acc)
				for name,param in self.model.named_parameters():
					print(name, param)

		print('Finished Training')

	def save_model(self):
		PATH = './fico_svm.pth'
		torch.save(self.model.state_dict(), PATH)



	def test_model(self, X_ts=None, y_ts=None):
		if (not self.model):
			raise ModelError("Train Model First")

		train_pred = self.model.predict(self.X_tr)
		test_pred = self.model.predict(self.X_test)

		acc_train = round((np.mean(train_pred.reshape(train_pred.shape[0],1) == self.y_tr)* 100),2)
		acc_test = round((np.mean(test_pred.reshape(test_pred.shape[0],1) == self.y_test)* 100),2)

		print("Training Accuracy:", acc_train, '%')
		print("Test Accuracy:", acc_test, '%')

	def __scaled_row(self,row):
	    scld = []
	    for k in range(row.shape[0]):
	        scld.append((row[k] - self.mean[k])/self.scale[k])
	    scld = np.array(scld)
	    
	    return np.array(scld)

	def run_model(self,sample):
		sample = self.__scaled_row(sample)
		if (not self.model):
			raise ModelError("Train Model First")

		result = self.model.predict_proba(sample.reshape(1, -1))

		return result[0][1]

	def run_model_data(self,data_set):
		if (not self.model):
			raise ModelError("Train Model First")

		for i in range(data_set.shape[0]):
			data_set[i] = self.__scaled_row(data_set[i])

		pred = self.model.predict(data_set)

		return pred 


 



