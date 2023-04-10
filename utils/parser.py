import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', nargs='?', default='MLP',
                        help='Specify a model type from {MLP, SVM}.')
    parser.add_argument('--dataset', nargs='?', default='adult',
                        help='Choose a dataset from {adult, FICO}')
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--model_path', nargs='?', default='data/out/weights/',
                        help='Store model path.')
    parser.add_argument('--out_dir', nargs='?', default='data/out/',
                        help='out path.')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1:Pretrain with stored models.')

    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')
    parser.add_argument('--regs', type=int, default=10,
                        help='Regularization for influence.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--device', nargs='?', default='cpu',
                        help='Choose a device from {cuda, cpu,mps}')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')

    # return parser.parse_args()
    return parser.parse_known_args()[0]
