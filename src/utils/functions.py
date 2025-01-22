import random
import time
from functools import wraps

import dgl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def record_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


def repeat_k_times(k, seed):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            accuracies = []
            for i in range(k):
                seed_everything(seed + i)
                accuracy = func(*args, **kwargs)
                accuracies.append(accuracy)

            if k > 1:
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                print(
                    f"Accuracy performance over {k} runs: {mean_accuracy * 100:.2f}_{{\\pm {std_accuracy * 100:.2f}}}")

        return wrapper

    return decorator


# todo: add torch GPU implementation if needed
def cal_metric(pred: torch.Tensor, target: torch.Tensor, metric: str) -> float:
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    if metric == 'accuracy':
        res = accuracy_score(target, pred)
    elif metric == 'micro-f1':
        res = f1_score(target, pred, average='micro')
    elif metric == 'macro-f1':
        res = f1_score(target, pred, average='macro')
    elif metric == 'auc':
        res = roc_auc_score(target, pred)
    elif metric == 'recall':
        res = recall_score(target, pred)
    else:
        raise NotImplementedError

    return res


if __name__ == '__main__':
    @record_time
    def test_time():
        time.sleep(2)
        return 0


    test_time()
