#! /usr/bin/env python3

import torch
from torch.autograd import grad
from IF.utils import display_progress
from svm.train import criterion

device =  "mps" if torch.backends.mps.is_available() else "cpu" 

def s_test(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    v = grad_z(z_test, t_test, model, gpu)
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.to(device), t.to(device)
            y = model(x)
            # weight = model.layer.weight.squeeze()
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
        display_progress("Calc. s_test recursions: ", i, recursion_depth)
    return h_estimate


def calc_loss(y, t):
    y = torch.nn.functional.sigmoid(y)
    loss = 1-y*t
    loss[loss<=0] = 0
    # loss = torch.nn.functional.nll_loss(
    #     y, t, weight=None, reduction='mean')
    return torch.sum(loss)


def grad_z(z, t, model, gpu=-1):
    model.eval()
    # initialize
    if gpu >= 0:
        z, t = z.to(device), t.to(device)
    y = model(z)
    # weight = model.layer.weight.squeeze()
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=True))


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
    return_grads = grad(elemwise_products, w,allow_unused=True )

    return return_grads
