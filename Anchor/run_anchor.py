import sys
import warnings

sys.path.append('./')
from anchor.anchor_tabular import *
from models.data_process import *
from models.run_MLP import load_model

warnings.simplefilter("ignore", UserWarning)
args = parse_args()
device = args.device


def find_anchor(dataset, explainer, predict_fn):
    anchors = {}
    # dataset_len = len(dataset)
    dataset_len = 1
    anchors['total'] = dataset_len
    for sample_id in range(dataset_len):
        x_test = dataset[sample_id]

        exp = explainer.explain_instance(x_test, predict_fn, threshold=0.95, max_anchor_size=5)
        exp_map = exp.exp_map
        anchors[sample_id] = {}
        anchors[sample_id]['feature'] = exp_map['names']
        anchors[sample_id]['precision'] = exp_map['precision']
        anchors[sample_id]['examples'] = exp_map['examples']  # 自己做数据处理

        # 打印示例
        # print('labelNames',explainer.class_names)
        # print(exp_map['names'])
        # print(exp_map['precision'])
        # print("Examples where the A.I. agent predicts covered_true: ",exp_map['examples'])
        display_progress(f"Calc. anchor test_id_{sample_id}: ", sample_id, dataset_len)

    outdir = Path(args.out_dir)
    outdir.mkdir(exist_ok=True, parents=True)
    save_json(anchors, outdir.joinpath(getFileName('anchors111', 'json')), overwrite_if_exists=True)


if __name__ == "__main__":
    model = load_model()
    x_dataset, target, encoder, categorical_names = load_adult_income_dataset()

    adult_process_names.insert(0, 'id')
    explainer = AnchorTabularExplainer(adult_target_value, adult_process_names, x_dataset,
                                       categorical_names=categorical_names)
    predict_fn = lambda x: model.predict_anchor(x, encoder)

    find_anchor(x_dataset, explainer, predict_fn)
