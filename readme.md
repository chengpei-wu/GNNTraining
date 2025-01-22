# A Library for Benchmark Training of Graph Neural Networks (GNNs) Based on [DGL](https://github.com/dmlc/dgl).
An alternative implementation of [https://github.com/LUOyk1999/tunedGNN](https://github.com/LUOyk1999/tunedGNN)

## Dependencies
- torch(2.2.1)
- DGL(2.0.0)

## Runing   
- Detailed training commands can be found in [run_node_benchmark.md](run_node_benchmark.md)
- The hyperparameter tuning script is available at [param_search.sh](param_search.sh)

## Results
- Node classification results for GCN, GAT, and GraphSAGE on homophilous, heterophilous, and large OGB graphs are reported in [benchmark_results.md](benchmark_results.md)

## Implementation features
- All GNN models and dataset processing pipelines are implemented using [DGL](https://github.com/dmlc/dgl).
- The GNNs model are structure-tuned follow [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://github.com/LUOyk1999/tunedGNN/tree/main), which demonstrates the strong capability of classical GNNs.
- A unified data processing framework is provided for loading and splitting datasets across homophilous, heterophilous, and OGB graphs.
- A consistent and flexible pipeline for training and evaluation is implemented, supporting various training configurations and arguments.