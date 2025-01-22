from copy import deepcopy

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.functions import cal_metric
from src.utils.train_utils import loss_mapping, optimizer_mapping


def evaluate(
        graph: dgl.DGLGraph, feats: torch.Tensor,
        labels: torch.Tensor, mask: torch.Tensor, model: nn.Module, metric: str,
        **kwargs
) -> float:
    model.eval()

    # todo: check whether it is right for other datasets
    if 'eval_fun' in kwargs:
        # if ogb evaluation function is provided
        eval_fun = kwargs['eval_fun']
        assert kwargs['eval_metric'] is not None, 'eval_metric should be provided when evaluate_fun is provided.'
        evaluate_metric = kwargs['eval_metric']
        with torch.no_grad():
            logits = model(graph, feats)
            if evaluate_metric == 'acc':
                _, pred = torch.max(logits[mask], 1, keepdim=True)
                labels = labels[mask].view(-1, 1)
            elif evaluate_metric == 'rocauc':
                pred = logits

            return eval_fun(
                {
                    'y_true': labels,
                    'y_pred': pred
                }
            )[evaluate_metric]
    else:
        with torch.no_grad():
            logits = model(graph, feats)
            logits = logits[mask]
            labels = labels[mask]
            if metric != 'auc':
                _, predicted = torch.max(logits, 1)
            else:
                if logits.shape[1] == 2:
                    logits = F.softmax(logits, dim=1)
                    predicted = logits[:, 1]
                elif logits.shape[1] > 2:
                    # multi-class classification as multiple binary classification
                    # triggered for ogbn-proteins dataset only (maybe extended to other datasets)
                    predicted = F.sigmoid(logits).view(-1)
                    labels = labels.view(-1)
            return cal_metric(predicted, labels, metric)


def train_loop(
        graph: dgl.DGLGraph, feats: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor,
        val_mask: torch.Tensor, test_mask: torch.Tensor, model: nn.Module, epochs: int, lr: float, metric: str,
        loss_fcn: str, optimizer: str, weight_decay: float, **kwargs
):
    loss_fcn = loss_mapping(loss_fcn)
    optimizer = optimizer_mapping(optimizer)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(epochs * 0.2)
    )

    # training loop
    with tqdm(total=epochs, desc='Epochs') as pbar:
        best_acc_test = 0
        best_acc_val = 0
        best_epoch = 0
        val_accs, test_accs = [], []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(graph, feats)

            loss = loss_fcn(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.item())

            val_acc = evaluate(graph, feats, labels, val_mask, model, metric, **kwargs)
            train_acc = evaluate(graph, feats, labels, train_mask, model, metric, **kwargs)
            test_acc = evaluate(graph, feats, labels, test_mask, model, metric, **kwargs)

            if val_acc > best_acc_val:
                best_acc_val = val_acc
                best_acc_test = test_acc
                best_epoch = epoch

            metrics = {
                "loss": loss.item(),
                f"train_{metric}": train_acc,
                f"val_{metric}": val_acc,
                f"test_{metric}": test_acc,
                f"best_val_{metric}[Epoch: {best_epoch}]": best_acc_val,
                f"best_test_{metric}[Epoch: {best_epoch}]": best_acc_test
            }

            val_accs.append(val_acc)
            test_accs.append(test_acc)

            pbar.set_postfix(metrics)
            pbar.update(1)
    best_index = np.argmax(val_accs)
    best_test_acc = test_accs[best_index]

    return best_test_acc


def train_node_classification(
        graph: dgl.DGLGraph, feats: torch.Tensor, labels: torch.Tensor,
        masks: tuple, model: nn.Module, epochs: int, lr: float, metric: str,
        loss_fun: str, optimizer: str, weight_decay: float, **kwargs
):
    train_mask = masks[0]
    val_mask = masks[1]
    test_mask = masks[2]

    if train_mask.ndim == 1:
        acc = train_loop(
            graph, feats, labels, train_mask, val_mask, test_mask,
            model, epochs, lr, metric, loss_fun, optimizer, weight_decay, **kwargs
        )
    else:
        accs = []
        for i in range(train_mask.shape[1]):
            model_ = deepcopy(model)
            if test_mask.ndim == 1:
                test_mask = test_mask.unsqueeze(1).expand(-1, train_mask.shape[1])
            acc = train_loop(
                graph, feats, labels, train_mask[:, i], val_mask[:, i], test_mask[:, i],
                model_, epochs, lr, metric, loss_fun, optimizer, weight_decay, **kwargs
            )
            accs.append(acc)
        acc = np.mean(accs)
        print(f"{metric} performance over {i} folds: {acc * 100:.2f}_{{\\pm {np.std(accs) * 100:.2f}}}")
    return acc
