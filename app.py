import copy
import warnings

import pandas as pd
from flask import Flask
from flask import request

from utils.helper import *
from utils.parser import parse_args

args = parse_args()
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告

np.random.seed(12345)

# ------- Initialize file ------- #
data_file = args.data_path + 'final_data.csv'
# pre_file = args.out_dir + f'pred_data_{args.model_type}_epoch{args.epoch}.csv'
pre_file = args.out_dir + getFileName('prediction', 'csv')
raw_data = json.loads(pd.read_csv(data_file, header=0).to_json(orient='records'))
pre_data = json.loads(pd.read_csv(pre_file, header=0).to_json(orient='records'))
raw_data_len = len(raw_data)
with open(args.out_dir + getFileName('influence', 'json'), 'r', encoding='utf-8') as fp:
    influenceData = json.load(fp)
    print('=====influence file read done')
fp.close()
with open(args.out_dir + getFileName('ianchors', 'json'), 'r', encoding='utf-8') as fp:
    anchorData = json.load(fp)
    print('=====anchor file read done')
fp.close()
with open(args.out_dir + getFileName('dice', 'json'), 'r', encoding='utf-8') as fp:
    diceData = json.load(fp)
    print('=====dice file read done')
fp.close()


# ------- Initialize Model ------- #
# 加载训练好的模型
# model = load_model(BASEDIR+'svm_FICO_500.pth')
# train_loader,test_loader,train_set,test_set= loader_data()
# model = load_model()


# ------ Help Function ------- #
def getSingleSample(idx):
    if idx < 0 or idx > raw_data_len:
        print(f'Please enter a correct number ,not {idx}')
        return None
    sample = copy.deepcopy(raw_data[idx])
    sample['prediction'] = adult_target_value[pre_data[idx]['prediction']]
    sample['category'] = pre_data[idx]['category']
    sample['percentage'] = pre_data[idx]['percentage']
    return sample


# ------ Initialize WebApp ------- #
app = Flask(__name__)


@app.route('/getPredData')
def getPredData():
    idx = -1
    try:
        idx = request.args.get('idx')
    except:
        return f"Please enter a sample number."

    response = {'total': len(pre_data)}

    if idx is not None:
        idx = int(idx)
        response['data'] = pre_data[idx]
        # response['data'] = pre_data.to_json(index = idx)
    else:
        response['data'] = pre_data

    return toJson(response)


@app.route('/getInstance')
def getInstance():
    idx = -10
    try:
        idx = int(request.args.get('params'))
    except:
        return f"Please enter a sample number in the range (1, ${raw_data_len})."

    if idx < 0 or idx > raw_data_len:
        return f"Please enter a sample number in the range (1, ${raw_data_len})."
    sample = getSingleSample(idx)
    response = {'id': idx, 'total': len(raw_data), 'sample': sample}
    return toJson(response)


@app.route('/getAnchor')
def getAnchor():
    try:
        idx = request.args.get('params')
    except:
        return f"Please enter a sample number in the range (1, ${raw_data_len})."

    if int(idx) < 0 or int(idx) > raw_data_len:
        return f"Please enter a sample number in the range (1, ${raw_data_len})."
    anchor = anchorData[idx]

    def getSampleCovered(covered):
        new_ct = []
        for ct_item in covered:
            new_ct_list = []
            for j in ct_item:
                tmp = getSingleSample(j)
                new_ct_list.append(tmp)
            new_ct.append(new_ct_list)
        return np.array(new_ct)

    response = {}
    response['feature'] = anchor['feature']
    response['precision'] = anchor['precision']
    response['coverage'] = anchor['coverage']
    response['covered_false'] = getSampleCovered(anchor['covered_false'])
    response['covered_true'] = getSampleCovered(anchor['covered_true'])
    return toJson(response)


# 获取相似数据-影响训练点
@app.route('/getInfluenceData')
def getInfluenceData():
    try:
        idx = request.args.get('params')
    except:
        return f"Please enter a sample number."
    # idx = request.args['params']
    total = influenceData['total']
    inData = influenceData[str(idx)]
    time = inData['time_calc_influence_s']
    influence = inData['influence']
    harmful = inData['harmful']
    helpful = inData['helpful']

    response = {'total': total, 'time': time, 'influence': [], 'harmful': [], 'helpful': [], 'max': inData['max'],
                'min': inData['min']}
    for i in range(len(harmful)):
        response['harmful'].append({
            'id': harmful[i],
            'value': influence[harmful[i]],
        })
        response['helpful'].append({
            'id': helpful[i],
            'value': influence[helpful[i]],
        })
    for i in range(len(influence)):
        response['influence'].append({
            'id': i,
            'value': influence[i],
            'sign': (1 if influence[i] else 0)
        })

    return toJson(response)


@app.route('/getDiceData')
def getDiceData():
    try:
        idx = request.args.get('params')
    except:
        return f"Please enter a sample number."

    inData = diceData[str(idx)]
    cfs_list = inData['cfs_list']
    pf = pd.DataFrame(cfs_list, columns=adult_process_names)
    cfs_list = pf.to_dict(orient='records')
    response = {'cfs_list': cfs_list}
    return toJson(response)


# @app.route('/runModel',methods=['GET', 'POST'])
# def runModel():
#     if request.method == 'POST':
#         print(request.json)
#         test_set = request.json
#         # model = load_model(save_path)
#         x_test = torch.Tensor(test_set).to(device)
#         y = model.pred(x_test).item()
#         y_prob = model(x_test).item()
#         print("Test result:\t{:.2%}\t{:.2%}".format(y,y_prob))

#         response = {}
#         response['predict'] = y
#         response['possible'] = y_prob
#         return toJson(response)


# ------- Run WebApp ------- #

if __name__ == '__main__':
    app.run(port=3001, host="0.0.0.0", debug=True, threaded=True)
# serve(app, host="0.0.0.0",port=3001)
