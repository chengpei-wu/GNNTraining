import torch
import torch.nn.functional as F


def loss_mapping(loss: str):
    if loss == 'cross_entropy':
        return F.cross_entropy
    elif loss == 'bce':
        return F.binary_cross_entropy
    elif loss == 'bce_with_logits':
        return F.binary_cross_entropy_with_logits
    elif loss == 'nll':
        return F.nll_loss
    elif loss == 'mse':
        return F.mse_loss
    elif loss == 'mae':
        return F.l1_loss
    else:
        raise NotImplementedError(loss)


def optimizer_mapping(optimizer: str):
    if optimizer == 'adam':
        return torch.optim.Adam
    elif optimizer == 'adamw':
        return torch.optim.AdamW
    elif optimizer == 'sgd':
        return torch.optim.SGD
    else:
        raise NotImplementedError(optimizer)
