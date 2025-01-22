from pprint import pprint

import torch
from ogb.nodeproppred import Evaluator

from src.backbones import model_mapping
from src.utils import load_node_dataset, train_node_classification
from src.utils.argparser import parse_args
from src.utils.functions import repeat_k_times

args = parse_args()
pprint(args)


@repeat_k_times(args.num_runs, args.random_seed)
def node_classification(args):
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    dataset_name = args.dataset
    model_name = args.model

    # print(f'Training {model_name} for {dataset_name} on {device} ...')

    graph, num_class = load_node_dataset(
        dataset_name=dataset_name,
        add_self_loop=args.self_loop,
        split=args.split,
        verbose=args.verbose
    )
    graph = graph.int().to(device)

    in_size = graph.ndata['feat'].shape[1]
    out_size = num_class

    gnn = model_mapping(model_name)

    gnn_args = {
        'in_size': in_size,
        'out_size': out_size,
        'hid_size': args.hid_size,
        'num_layers': args.num_layers,
        'pooling': args.pooling,
        'drop_out': args.drop_out,
        'in_linear': args.in_linear,
        'out_linear': args.out_linear,
        'is_res': args.is_res,
        'is_ln': args.is_ln,
        'is_bn': args.is_bn,
        'is_jk': args.is_jk
    }
    if args.model == 'GAT':
        gnn_args.setdefault('num_heads', args.gat_heads)
        gnn_args.setdefault('atten_drop', args.gat_atten_drop)

    model = gnn(
        **gnn_args
    ).to(device)

    # print(model)

    masks = graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask']

    train_args = {
        'graph': graph,
        'feats': graph.ndata['feat'],
        'labels': graph.ndata['label'],
        'masks': masks,
        'model': model,
        'epochs': args.epochs,
        'lr': args.lr,
        'metric': args.validation_metric,
        'loss_fun': args.loss_fun,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay
    }

    if args.use_ogb_eval:
        assert dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag',
                                'ogbn-papers100M'], 'only ogb dataset provides evaluator.'

        evaluator = Evaluator(name=dataset_name)
        train_args.setdefault('eval_fun', evaluator.eval)
        train_args.setdefault('eval_metric', evaluator.eval_metric)
        train_args.setdefault('metric', evaluator.eval_metric)

    acc = train_node_classification(
        **train_args
    )

    return acc


if __name__ == '__main__':
    node_classification(args)
