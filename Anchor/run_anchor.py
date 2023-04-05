import sys
import os
sys.path.append(os.curdir)

import pandas as pd

from utils.parser import *
from utils.helper import *
from models.data_process import *
from models.run_MLP import MLP,load_model
from anchor.anchor_tabular import *

import warnings
warnings.simplefilter("ignore", UserWarning)

args = parse_args()
device = args.device

def find_anchor(dataset,explainer,predict_fn):

    anchors = {}
    # dataset_len = len(dataset)
    dataset_len = 10
    for sample_id in range(dataset_len):
        x_test = dataset[sample_id]

        exp = explainer.explain_instance(x_test,predict_fn, threshold=0.95,max_anchor_size=5)
        exp_map = exp.exp_map
        anchors[sample_id] = {}
        anchors[sample_id]['feature'] = exp_map['names']
        anchors[sample_id]['precision'] = exp_map['precision']
        anchors[sample_id]['examples'] = exp_map['examples']

        # 打印示例
        # print('labelNames',explainer.class_names)
        # print(exp_map['names'])
        # print(exp_map['precision'])
        # print("Examples where the A.I. agent predicts covered_true: ",exp_map['examples'])
        display_progress(f"Calc. anchor test_id_{sample_id}: ", sample_id, dataset_len)


    outdir = Path(args.out_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    anchors_path = outdir.joinpath(f"anchors_{args.model_type}_{dataset_len}.json")
    save_json(anchors, anchors_path,overwrite_if_exists=True)

if __name__ == "__main__":
    model = load_model()
    x_dataset,target,encoder,categorical_names = load_adult_income_dataset()
    # train_dataset, test_dataset, y_train, y_test = train_test_split(x_dataset,
    #                                                             target,
    #                                                             test_size=0.2,
    #                                                             random_state=0,
    #                                                             stratify=target)
    
    explainer = AnchorTabularExplainer(adult_target_value, adult_process_names, x_dataset, categorical_names = categorical_names)
    predict_fn = lambda x: model.predict_anchor(x,encoder)

    
    find_anchor(x_dataset,explainer,predict_fn)

