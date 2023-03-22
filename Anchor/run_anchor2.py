import sys
import os
sys.path.append(os.curdir)

from utils.parser import *
from models.data_process import *
from models.run_MLP import *
from anchor import anchor_tabular

args = parse_args()
device = args.device

if __name__ == "__main__":
    test_dataset = Adult_data(mode = 'test')
    sample_id = 0
    x_test_idx,y_test_idx = test_dataset[sample_id]

    model = MLP().to(device)
    ckpt = torch.load(args.model_path+f'MPL_{args.epoch}.pth', map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()

    pred = model.predict_single(x_test_idx)
    print('Prediction: ',adult_target_value[pred],'\tlabel: ',pred )
    x_test_idx = x_test_idx.detach().numpy()


    train_dataset = Adult_data(mode = 'train',tensor = False)
    x_train= train_dataset.dataset
    y_train = train_dataset.target

    print(x_train.shape)
    print(len(adult_oneHot_names[:-1]))

    explainer = anchor_tabular.AnchorTabularExplainer([b'<=50K', b'>50K'], adult_oneHot_names[:-1], x_train, categorical_names = {})

    exp = explainer.explain_instance(x_test_idx, model.predict_detach, threshold=0.95)