from src.backbones.gnns import GAT as GAT, GCN as GCN, GIN as GIN, SAGE as SAGE

__all__ = ['SAGE', 'GAT', 'GIN', 'model_mapping']


def model_mapping(model: str):
    if model == 'GCN':
        return GCN
    elif model == 'GAT':
        return GAT
    elif model == 'GIN':
        return GIN
    elif model == 'SAGE':
        return SAGE
    else:
        raise NotImplementedError(model)
