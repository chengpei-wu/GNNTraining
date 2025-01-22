import dgl
import torch
from dgl.data import (AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, AmazonRatingsDataset, ChameleonDataset, CiteseerGraphDataset, CoauthorCSDataset, CoauthorPhysicsDataset,
                      CoraGraphDataset, FlickrDataset, MinesweeperDataset, PubmedGraphDataset, QuestionsDataset, RedditDataset, RomanEmpireDataset,
                      SquirrelDataset, TolokersDataset, WikiCSDataset)
from ogb.nodeproppred import DglNodePropPredDataset

dataset_mapping = {
    'cora': CoraGraphDataset,
    'citeseer': CiteseerGraphDataset,
    'pubmed': PubmedGraphDataset,
    'am-photo': AmazonCoBuyPhotoDataset,
    'am-computer': AmazonCoBuyComputerDataset,
    'co-cs': CoauthorCSDataset,
    'co-physics': CoauthorPhysicsDataset,
    'wiki-cs': WikiCSDataset,
    'reddit': RedditDataset,
    'roman-empire': RomanEmpireDataset,
    'amazon-ratings': AmazonRatingsDataset,
    'minesweeper': MinesweeperDataset,
    'tolokers': TolokersDataset,
    'questions': QuestionsDataset,
    'squirrel': SquirrelDataset,
    'chameleon': ChameleonDataset,
    'flickr': FlickrDataset
}


def random_ratio_split(
        num_samples: int,
        train_size: float = 0.1, valid_size: float = 0.1, test_size: float = 0.8
) -> dict:
    assert train_size + valid_size + test_size == 1.0, 'train_size + valid_size + test_size must equal to 1.0'
    indices = torch.randperm(num_samples)

    num_train = int(train_size * num_samples)
    num_test = int(test_size * num_samples)

    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    train_mask[indices[:num_train]] = True
    test_mask[indices[num_train:num_train + num_test]] = True
    val_mask[indices[num_train + num_test:]] = True

    return {'train_mask': train_mask, 'test_mask': test_mask, 'val_mask': val_mask}


def random_class_num_split(num_samples: int, labels: torch.Tensor, num_train: int, num_val: int, num_test: int) -> dict:
    unique_labels = torch.unique(labels)

    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        perm = torch.randperm(len(label_indices))
        train_indices = label_indices[perm[:num_train]]
        train_mask[train_indices] = True

    remaining_indices = torch.where(~train_mask)[0]

    perm = torch.randperm(len(remaining_indices))
    val_indices = remaining_indices[perm[:num_val]]
    test_indices = remaining_indices[perm[num_val:num_val + num_test]]

    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}


def few_shot_split(num_samples: int, labels: torch.Tensor, num_train: int, num_val: int) -> dict:
    unique_labels = torch.unique(labels)
    train_mask = torch.zeros(num_samples, dtype=torch.bool)
    val_mask = torch.zeros(num_samples, dtype=torch.bool)
    test_mask = torch.zeros(num_samples, dtype=torch.bool)

    for label in unique_labels:
        label_indices = torch.where(labels == label)[0]
        perm = torch.randperm(len(label_indices))
        train_indices = label_indices[perm[:num_train]]
        val_indices = label_indices[perm[num_train:num_train + num_val]]
        test_indices = label_indices[perm[num_train + num_val:]]

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

    return {'train_mask': train_mask, 'test_mask': test_mask, 'val_mask': val_mask}


def load_node_dataset(dataset_name: str, add_self_loop: bool, split: str = 'public', verbose=False):
    # load ogb datasets
    if dataset_name in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag', 'ogbn-papers100M']:
        dataset = DglNodePropPredDataset(name=dataset_name)
        graph, label = dataset[0]

        split_idx = dataset.get_idx_split()
        num_nodes = graph.num_nodes()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

        if dataset_name == 'ogbn-proteins':
            # For ogbn-proteins, there is lack of node features.
            # we use the average edge features of incoming edges as node features.
            graph.update_all(
                dgl.function.copy_e('feat', 'm'),
                dgl.function.mean('m', 'feat')
            )
            label = label.float()

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        valid_mask[valid_idx] = True
        test_mask[test_idx] = True

        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = valid_mask
        graph.ndata['test_mask'] = test_mask

        graph.ndata['label'] = label.squeeze()
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)

    # load dgl datasets
    else:
        if add_self_loop:
            self_loop = dgl.AddSelfLoop()
        else:
            self_loop = None

        if dataset_name in dataset_mapping:
            dataset = dataset_mapping[dataset_name](transform=self_loop, verbose=verbose)
        else:
            raise NotImplementedError(f'Dataset {dataset_name} not implemented')

        graph = dataset[0]

    num_nodes = graph.num_nodes()
    if split == 'public':
        graph.ndata['train_mask'] = graph.ndata['train_mask'].bool()
        graph.ndata['test_mask'] = graph.ndata['test_mask'].bool()
        graph.ndata['val_mask'] = graph.ndata['val_mask'].bool()
    elif 'random-r' in split:
        train_size, valid_size, test_size = float(split.split('-')[2]), float(split.split('-')[3]), float(
            split.split('-')[4])
        masks = random_ratio_split(
            num_samples=num_nodes,
            train_size=train_size, valid_size=valid_size, test_size=test_size
        )
        graph.ndata['train_mask'] = masks['train_mask']
        graph.ndata['test_mask'] = masks['test_mask']
        graph.ndata['val_mask'] = masks['val_mask']
    elif 'random-c' in split:
        train_size, valid_size, test_size = int(split.split('-')[2]), int(split.split('-')[3]), int(
            split.split('-')[4])
        masks = random_class_num_split(
            num_samples=num_nodes, labels=graph.ndata['label'],
            num_train=train_size, num_val=valid_size, num_test=test_size
        )
        graph.ndata['train_mask'] = masks['train_mask']
        graph.ndata['test_mask'] = masks['test_mask']
        graph.ndata['val_mask'] = masks['val_mask']
    elif '-shot' in split:
        k = int(split.split('-')[0])
        masks = few_shot_split(num_samples=num_nodes, labels=graph.ndata['label'], num_train=k, num_val=k)
        graph.ndata['train_mask'] = masks['train_mask']
        graph.ndata['test_mask'] = masks['test_mask']
        graph.ndata['val_mask'] = masks['val_mask']
    else:
        raise NotImplementedError(f'{split} split has not implemented')

    num_class = dataset.num_classes

    if dataset_name == 'ogbn-proteins':
        num_class = graph.ndata['label'].shape[1]

    return graph, num_class


if __name__ == '__main__':
    # load_node_dataset('ogbn-arxiv', add_self_loop=True, split='public', verbose=True)
    # load_graph_dataset('MUTAG', add_self_loop=True, verbose=False, batch_size=16, split='5-shot')
    # load_graph_dataset('ogbg-molhiv', add_self_loop=True, verbose=False, batch_size=16, split='5-shot')
    # load_link_dataset('cora', add_self_loop=True, split='20-shot', verbose=False)
    # train_g, val_g, test_g, num_class = load_link_dataset('cora', add_self_loop=True, split='random-0.1-0.1-0.8',
    #     verbose=False)
    load_node_dataset('ogbn-proteins', add_self_loop=True, split='public', verbose=True)
