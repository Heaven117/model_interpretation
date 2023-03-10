import numpy as np
import torch

import os
import sys
sys.path.append(os.curdir)

from svm.data_process import load_data,loader_data
from svm.train import load_model,train
from svm.SVM_model import ModelError
from utils import get_default_config

def evaluate_data_set(data):
	no_features = data.shape[1]
	avg_list = []
	std_list = []
	for i in range(no_features):
		current_col = data[:,i].flatten()
		std_list.append(np.std(current_col))
		avg_list.append(np.mean(current_col))
		  
	return avg_list, std_list

def perturb_special(min_val,max_val,avg,std,no_val):
	new_col = np.random.normal(avg, std, no_val)
	# Note: these functions have poor time complexity
	np.place(new_col,new_col < min_val, min_val)
	np.place(new_col,new_col > max_val, max_val)
	new_col = new_col.round(0)
	return new_col
	

def find_anchors(model, train_loader, sample_id, no_val):
	x_sample, y_sample = train_loader.dataset[sample_id]
	data_set = train_loader.dataset.data['X'].cpu().numpy()
	# Account for the special categorical columns
	special_cols = [9,10]
	
	features = len(x_sample)
	avg_list, std_list = evaluate_data_set(data_set)

	# Precision Treshold
	treshold = 0.95
	
	# Identify original result from sample
	initial_percentage = model(x_sample)
	decision = torch.round(initial_percentage)
	# decision = np.round(initial_percentage,0)

	# Create empty mask 
	mask = np.zeros(features)
	
	# Allows tracking the path
	locked = []
	
	# Iterations allowed
	iterations = 4

	# Setting random seed
	np.random.seed(150)

	
	while (iterations > 0):
		# Retains best result and the corresponding index
		max_ind = (0,0)

		# Assign column that is being tested
		for test_col in range(features):
			new_data = np.empty([features, no_val]).astype('float32')
			# new_data.dtype = 'float32'
			# new_data = torch.zeros([features, no_val])


			# Perturb data
			for ind in range(features):
				if (ind == test_col) or (ind in locked):
					new_data[ind] = np.array(np.repeat(x_sample[ind],no_val))
				else:
					if (ind in special_cols):
						new_data[ind] = perturb_special(0,7,avg_list[ind],std_list[ind],no_val)
					else:
						new_data[ind] = np.random.normal(avg_list[ind], std_list[ind], no_val)

			
			new_data = new_data.transpose()

			# Run Model 
			new_data = torch.from_numpy(new_data)
			pred,_ = model.pred_prod(new_data)
			_acc = torch.mean(pred == decision)
			acc = (np.mean(pred == decision))
			
			if (acc > max_ind[0]):
				max_ind = (acc,test_col)
				

		locked.append(max_ind[1])
			
		for n in locked:
			mask[n] = 1
			
		if (max_ind[0] >= treshold):
			return mask
		iterations -= 1
		
	print("!!! No anchors found !!!")
	return None


if __name__ == "__main__":
	model_config,IF_config = get_default_config()
	X_train,X_test,y_train,y_test= load_data()
	train_loader,test_loader= loader_data()

	save_path = model_config['save_path']
	if(os.path.exists(save_path)):
		model = load_model(save_path)
	else:
		raise ModelError("Train Model First")

	sample_id = 0
	find_anchors(model,train_loader,sample_id,100)
