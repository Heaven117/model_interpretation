import numpy as np
import torch
from anchor import anchor_tabular

import os
import sys
sys.path.append(os.curdir)

from svm.data_process import load_data, loader_data
from svm.run_model import load_model,test
from svm.SVM_model import ModelError
from utils import ft_names,model_config,IF_config ,save_path

# 简易版anchor
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
    data_set = train_loader.dataset.tensor_data['X'].cpu().numpy()
    # Account for the special categorical columns
    special_cols = [9,10]
    
    features = len(x_sample)
    avg_list, std_list = evaluate_data_set(data_set)

    # Precision Treshold
    treshold = 0.95
    
    # Identify original result from sample
    decision = model.pred(x_sample)

    # Create empty mask 
    mask = np.zeros(features)
    
    # Allows tracking the path
    locked = []
    
    # Iterations allowed
    iterations = 10

    # Setting random seed
    np.random.seed(150)

    
    while (iterations > 0):
        # Retains best result and the corresponding index
        max_ind = (0,0)

        # Assign column that is being tested
        for test_col in range(features):
            new_data = np.empty([features, no_val]).astype('float32')

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
            pred = model.batch_pred(new_data)
            acc = (pred == decision).sum().item() / new_data.shape[0]
            
            if (acc > max_ind[0]):
                max_ind = (acc,test_col)
                

        locked.append(max_ind[1])
            
        for n in locked:
            mask[n] = 1
            
        if (max_ind[0] >= treshold):
            print(f'id: {sample_id}\t anchor: {mask}')
            return mask
        iterations -= 1

    print(f'id: {sample_id}\t anchor: !!! No found !!!')
    return None

# 官方anchor
def anchors_tabular(model,idx,train_set,test_set):
    X_train,y_train = train_set.getScaleData()
    X_test,y_test = test_set.getScaleData()
    x_sample = X_train[idx]

    explainer1 = anchor_tabular.AnchorTabularExplainer([0,1], ft_names, X_train, categorical_names = {})
    # explainer.fit(X_train.values, y_train, X_val.values, y_val, discretizer='quartile')

    # decision = model.pred(x_sample)
    # if decision < 0 :
    #     decision = 0
    # else:
    #     decision = 1
    # print('Prediction: ', explainer.class_names[decision])
    # exp = explainer.explain_instance(x_sample.detach().numpy(), model.pred_numpy, threshold=0.95)
    exp = explainer1.explain_instance(x_sample, model.pred_numpy, threshold=0.95)

    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    # Get test examples where the anchora pplies
    X_train_row ,_ = train_set.getRawData()
    # todo 有问题，连续值不能绝对相等
    fit_anchor = np.where(np.all(X_train_row[:, exp.features()] == X_train_row[idx][exp.features()], axis=1))[0]
    print('Anchor test precision: %.2f' % (np.mean(model.pred_numpy(X_train[fit_anchor]) == model.pred_numpy(X_train[idx].reshape(1, -1)))))
    print('Anchor test coverage: %f' % (fit_anchor.shape[0] / float(X_train.shape[0])))

if __name__ == "__main__":
    train_loader,test_loader,train_set,test_set= loader_data()
    test(train_set,test_set)
  
    if(os.path.exists(save_path)):
        model = load_model(save_path)
    else:
        raise ModelError("Train Model First")

    # sample_id = 1
    for sample_id in range(5):
        find_anchors(model,train_loader,sample_id,100)
        anchors_tabular(model,sample_id,train_set,test_set)

