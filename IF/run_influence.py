import os
import sys
sys.path.append(os.curdir)

import torch
import time
import numpy as np
from torch.autograd import grad

from svm.data_process import loader_data
from svm.run_model import load_model,train
from pathlib import Path
from utils import save_json, display_progress,model_config,IF_config,save_path
device = model_config['device']
mode_name = model_config['mode_name']
epoch = model_config['epoch']

"""
Hessian * vector
Arguments:
    y: loss func 满足二阶可导
    w: params
    v: vector will be multiplied with the Hessian, same shape as w
Returns:
    return_grads: product of Hessian and v.
"""
def hvp(y, w, v):
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    # todo 第二次求导在mac有问题
    return_grads = grad(elemwise_products, w, retain_graph=True)

    return return_grads

"""
损失函数必须二阶可导
hinge loss不可微,因此采用smoothHinge
todo 
"""
def calc_loss(y, output):
    temp = 0.001
    s = y*output
    exponents = (1-s)/temp
    max_elems = torch.maximum(exponents, torch.zeros_like(exponents))
    loss =  temp * (max_elems + torch.log(
        torch.exp(exponents - max_elems) + 
        torch.exp(torch.zeros_like(exponents) - max_elems)))
    
    # loss = 1-y*output
    # loss[loss<=0] = 0
    return torch.sum(loss)

"""
计算梯度。每个train data都应该计算一个grad_z。
"""
def grad_z(z, t, model):
    model.eval()
    z, t = z.to(device), t.to(device)
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=True))

"""
s_test is the Inverse Hessian Vector Product.(HVP)
s_test = invHessian * nabla(Loss(test, model params))
L_up,loss = s_test * grad_z

Arguments:
    z: test data points
    t: test data labels
    damp: float, dampening factor 阻尼因子
    scale: float, scaling factor  比例因子
    recursion_depth: 迭代次数,多次迭代求平均
Returns:
    h_estimate: list of torch tensors, s_test
"""
def s_test(z, t, model, train_loader, damp=0.01, scale=25.0, recursion_depth=5000):
    v = grad_z(z, t, model)
    h_estimate = v.copy()

    for i in range(recursion_depth):
        for x, t in train_loader:
            x, t = x.to(device), t.to(device)
            y = model(x)
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate 递归计算
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


# 计算所有train data对单个test data的影响
def calc_influence_single(model, train_loader, test_loader,test_id_num, recursion_depth,  s_test_vec=None):
    if not s_test_vec:
        z_test, t_test = test_loader.dataset[test_id_num]
        s_test_vec = s_test(z_test, t_test,model,train_loader, recursion_depth=recursion_depth)
    
    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        z, t = train_loader.dataset[i]
        grad_z_vec = grad_z(z, t, model)
        tmp_influence = -sum(
            [
                # torch.sum(k * j).data.cpu().numpy()
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress(f"Calc. influence test_id_{test_id_num}: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num

# 计算所有test data的影响
def calc_main(config, model,train_loader,test_loader,start=0):
    outdir = Path(config['out_path'])
    outdir.mkdir(exist_ok=True, parents=True)

    # todo test设置1
    test_dataset_iter_len = 20
    # test_dataset_iter_len = len(test_loader.dataset)
    influences = {}
    last = start
    for i in range(start,test_dataset_iter_len):
        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(model,train_loader, test_loader, test_id_num=i, recursion_depth=config['recursion_depth'])
        end_time = time.time()
        
        influences['total'] = i + 1
        influences[str(i)] = {}
        influences[str(i)]['time_calc_influence_s'] = end_time - start_time
        infl = [x.cpu().numpy().tolist() for x in influence]
        influences[str(i)]['influence'] = infl
        influences[str(i)]['harmful'] = harmful[:500]
        influences[str(i)]['helpful'] = helpful[:500]

        if(i!=0 and i % 99 == 0):
            influences_path = outdir.joinpath(f"influence_tmp_{last}-{i}.json")
            save_json(influences, influences_path,overwrite_if_exists=True)
            last = i

    influences_path = outdir.joinpath(f"influence_{mode_name}_{epoch}-{i}.json")
    save_json(influences, influences_path,overwrite_if_exists=True)

if __name__ == "__main__":
    batch_size = model_config['batch_size']

    train_loader,test_loader,train_set,test_set= loader_data(batch_size)

    if(os.path.exists(save_path)):
        model = load_model(save_path)
    else:
        model = train(train_loader,test_loader)

    calc_main(IF_config, model,train_loader,test_loader,start=0)
   
