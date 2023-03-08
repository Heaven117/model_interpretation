import time
import torch
import numpy as np

from pathlib import Path
from IF.IF_svm import s_test, grad_z
from IF.utils import save_json, display_progress

# 不使用科学计数法
# np.set_printoptions(precision=4, suppress=True)
# torch.set_printoptions(precision=4, sci_mode=False)

# Calculates s_test for a single test image taking into account the whole training dataset.
# s_test = invHessian * nabla(Loss(test_img, model params))
def calc_s_test_single(model, z_test, t_test, train_loader, gpu=-1, damp=0.01, scale=25, recursion_depth=5000, r=1):
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(z_test, t_test, model, train_loader,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        display_progress("Averaging r-times: ", i, r)

    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec
     
# Calculates the influences of all training data points on a single test dataset.
def calc_influence_single(model, train_loader, test_loader,test_id_num, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False):
    if not s_test_vec:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test]) #todo 可能无用
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test,train_loader, gpu, recursion_depth=recursion_depth, r=r)
    
    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        tmp_influence = -sum(
            [
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist(), test_id_num


def calc_main(config, model,train_loader,test_loader):
    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    test_dataset_iter_len = len(test_loader.dataset)
    influences = {}
    for i in range(test_dataset_iter_len):
        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            model,train_loader, test_loader, test_id_num=i, gpu=config['gpu'],
            recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        end_time = time.time()

        influences[str(i)] = {}
        influences[str(i)]['time_calc_influence_s'] = end_time - start_time
        infl = [x.cpu().numpy().tolist() for x in influence]
        influences[str(i)]['influence'] = infl
        influences[str(i)]['harmful'] = harmful[:500]
        influences[str(i)]['helpful'] = helpful[:500]

        tmp_influences_path = outdir.joinpath(f"influence_res_tmp-{config['mode_name']}_{i}.json")
        save_json(influences, tmp_influences_path,overwrite_if_exists=True)
        display_progress("Test samples processed: ", i, test_dataset_iter_len)

