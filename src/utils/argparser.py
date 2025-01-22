import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # for dataset
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--self_loop', action='store_true', help='add self-loop')
    parser.add_argument('--verbose', action='store_true', help='do not print dataset info.')

    # for general training
    parser.add_argument('--random_seed', type=int, default='42')
    parser.add_argument('--no_cuda', action='store_true', help='do not use cuda devices.')
    parser.add_argument('--device', type=int, default='0')
    parser.add_argument('--epochs', type=int, default='200')
    parser.add_argument('--lr', type=float, default='1e-2')
    parser.add_argument('--num_runs', type=int, default='1')
    parser.add_argument('--loss_fun', type=str, default='cross_entropy')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default='5e-4')
    parser.add_argument('--use_ogb_eval', action='store_true', help='do not use ogb evaluator by default.')

    # for general model setting
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--num_layers', type=int, default='2')
    parser.add_argument('--hid_size', type=int, default='16')
    parser.add_argument('--drop_out', type=float, default='0.5')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--pooling', type=str, default=None)
    parser.add_argument('--in_linear', action='store_true', help='do not use input linear transformation by default.')
    parser.add_argument('--out_linear', action='store_true', help='do not use output linear transformation by default.')
    parser.add_argument('--is_res', action='store_true', help='do not use residual connection by default.')
    parser.add_argument('--is_ln', action='store_true', help='do not use layer normalization by default.')
    parser.add_argument('--is_bn', action='store_true', help='do not use batch normalization by default.')
    parser.add_argument('--is_jk', action='store_true', help='do not use jumping knowledge by default.')

    # for specific model/task setting
    parser.add_argument('--validation_metric', type=str, default='accuracy')
    parser.add_argument('--gat_heads', type=int, default=1)
    parser.add_argument('--gat_atten_drop', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default='256', help='for graph classification training.')

    args = parser.parse_args()
    return args
