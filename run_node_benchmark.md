
# Settings

For each dataset, the first command represents the results of the standard GNN (i.e., without significant structural
tuning such as residual connections, normalization, jumping knowledge, or input-output linear transformations).  
The second command reflects the results with structural tuning, largely based on the
paper: [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://github.com/LUOyk1999/tunedGNN/tree/main).

## GCN

```bash

# GCN over homophilous datasets

# Cora
python node_classification.py --dataset cora --split public --model GCN --self_loop --num_layers 2 --hid_size 64 --num_runs 5 --epochs 500 --drop_out 0.6 --lr 1e-2

# CiteSeer
python node_classification.py --dataset citeseer --split public --model GCN --self_loop --num_layers 2 --hid_size 512 --num_runs 5 --epochs 400 --drop_out 0.5 --lr 1e-3

# Pumbed
python node_classification.py --dataset pubmed --split public --model GCN --self_loop --num_layers 2 --hid_size 256 --num_runs 5 --epochs 400 --drop_out 0.7 --lr 5e-3

# Amazon-Computer
python node_classification.py --dataset am-computer --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 4 --hid_size 512 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset am-computer --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 3 --hid_size 512 --num_runs 5 --epochs 1000 --lr 1e-3 --drop_out 0.5 --is_ln --out_linear --device 1 

# Amazon-Photo
python node_classification.py --dataset am-photo --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 4 --hid_size 128 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset am-photo --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 6 --hid_size 256 --num_runs 5 --epochs 1000 --lr 1e-3 --drop_out 0.5 --is_ln --is_res --out_linear --device 1

# Coauthor-CS
python node_classification.py --dataset co-cs --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 2 --hid_size 16 --num_runs 5 --epochs 400 --lr 1e-3
python node_classification.py --dataset co-cs --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 2 --hid_size 64 --num_runs 5 --epochs 1500 --lr 1e-3 --drop_out 0.3 --is_ln --is_res --out_linear --device 1

# Coauthor-Physics
python node_classification.py --dataset co-physics --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 2 --hid_size 16 --num_runs 5 --epochs 400 --lr 1e-3
python node_classification.py --dataset co-physics --split random-r-0.6-0.2-0.2 --model GCN --self_loop --num_layers 2 --hid_size 64 --num_runs 5 --epochs 1500 --lr 1e-3 --drop_out 0.3 --is_ln --is_res --out_linear --device 1

# wiki-cs
python node_classification.py --dataset wiki-cs --split public --model GCN --self_loop --num_layers 2 --hid_size 16 --epochs 2000 --lr 1e-3
python node_classification.py --dataset wiki-cs --split public --model GCN --self_loop --num_layers 3 --hid_size 512 --epochs 1500 --lr 1e-3 --drop_out 0.5 --is_ln --out_linear --device 1

# GCN over heterophilous datasets

# Amazon-Ratings
python node_classification.py --dataset amazon-ratings --split public --model GCN --self_loop --num_layers 4 --hid_size 128 --epochs 2000 --lr 1e-2
python node_classification.py --dataset amazon-ratings --split public --model GCN --self_loop --num_layers 4 --hid_size 512 --epochs 600 --lr 1e-3 --drop_out 0.5 --is_bn --out_linear --is_res --device 1

# Roman-Empire (residual connection is crucial)
python node_classification.py --dataset roman-empire --split public --model GCN --self_loop --num_layers 2 --hid_size 64 --epochs 2000 --lr 1e-2
python node_classification.py --dataset roman-empire --split public --model GCN --self_loop --num_layers 8 --hid_size 512 --epochs 2500 --lr 1e-3 --drop_out 0.5 --in_linear --out_linear --is_bn --is_res --device 1

# Minesweeper
python node_classification.py --dataset minesweeper --split public --model GCN --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc 
python node_classification.py --dataset minesweeper --split public --model GCN --self_loop --num_layers 12 --hid_size 64 --epochs 800 --lr 1e-2 --drop_out 0.2 --validation_metric auc --is_bn --out_linear --is_res --device 1

# Questions
python node_classification.py --dataset questions --split public --model GCN --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc
python node_classification.py --dataset questions --split public --model GCN --self_loop --num_layers 9 --hid_size 512 --epochs 1500 --lr 3e-5 --drop_out 0.3 --validation_metric auc --in_linear --out_linear --is_res --device 1

# Tolokers
python node_classification.py --dataset tolokers --split public --model GCN --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc
python node_classification.py --dataset tolokers --split public --model GCN --self_loop --num_layers 9 --hid_size 512 --epochs 1500 --lr 3e-5 --drop_out 0.3 --validation_metric auc --in_linear --out_linear --is_res --device 1

# GCN over OGB large graphs

# ogbn-arxiv (without batch normalization)
python node_classification.py --dataset ogbn-arxiv --split public --model GCN --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 800 --lr 1e-2 --use_ogb_eval
python node_classification.py --dataset ogbn-arxiv --split public --model GCN --self_loop --num_layers 5 --hid_size 512 --num_runs 5 --epochs 2000 --lr 5e-4 --drop_out 0.5 --use_ogb_eval --is_bn --out_linear --is_res --device 1

# ogbn-products (without batch normalization)
python node_classification.py --dataset ogbn-products --split public --model GCN --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 300 --lr 1e-2 --use_ogb_eval
python node_classification.py --dataset ogbn-products --split public --model GCN --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 300 --lr 3e-3 --drop_out 0.5 --use_ogb_eval --is_ln --out_linear --device 0 

# ogbn-proteins (without batch normalization)
python node_classification.py --dataset ogbn-proteins --split public --model GCN --self_loop --num_layers 2 --hid_size 16 --num_runs 5 --epochs 300 --lr 1e-2 --loss_fun bce_with_logits --use_ogb_eval
python node_classification.py --dataset ogbn-proteins --split public --model GCN --self_loop --num_layers 3 --hid_size 512 --num_runs 5 --epochs 100 --lr 1e-2 --drop_out 0.25 --loss_fun bce_with_logits --use_ogb_eval --out_linear --is_res --is_bn

# the others parameters that not explicitly assigned are follow the default configuration.
```

## GAT

```bash
# GAT for homophilous datasets

# Cora
python node_classification.py --dataset cora --split public --self_loop --model GAT --num_layers 2 --hid_size 512 --num_runs 5 --epochs 1500 --lr 1e-2 --drop_out 0.7 --gat_atten_drop 0.7

# CiteSeer
python node_classification.py --dataset citeseer --model GAT --self_loop --split public --num_layers 2 --hid_size 512 --drop_out 0.2 --gat_atten_drop 0.6 --lr 1e-2 --num_runs 5 --epochs 400 --device 1

# Pumbed
python node_classification.py --dataset pubmed --split public --self_loop --model GAT --num_layers 2 --hid_size 512 --num_runs 5 --epochs 1000 --lr 5e-3 --drop_out 0.7 --gat_atten_drop 0.7 --out_linear --is_res

# Amazon-Computer
python node_classification.py --dataset am-computer --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 128 --num_runs 5 --epochs 1500 --lr 1e-3
python node_classification.py --dataset am-computer --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 128 --num_runs 5 --epochs 1500 --lr 1e-3 --gat_heads 4 --drop_out 0.6 --gat_atten_drop 0.6 --is_ln --is_res --out_linear --device 1

# Amazon-Photo
python node_classification.py --dataset am-photo --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 128 --gat_heads 1 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset am-photo --split random-r-0.6-0.2-0.2 --model GAT --num_layers 3 --hid_size 64 --gat_heads 4 --num_runs 5 --epochs 1000 --lr 1e-3 --drop_out 0.8 --gat_atten_drop 0 --is_ln --out_linear --is_res

# Coauthor-CS
python node_classification.py --dataset co-cs --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 16 --gat_heads 1 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset co-cs --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 256 --gat_heads 4 --num_runs 5 --epochs 1500 --lr 1e-3 --drop_out 0.7 --gat_atten_drop 0 --is_bn --is_res --out_linear --device 1

# Coauthor-Physics
python node_classification.py --dataset co-physics --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 16 --gat_heads 1 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset co-physics --split random-r-0.6-0.2-0.2 --model GAT --num_layers 2 --hid_size 256 --gat_heads 4 --num_runs 5 --epochs 1500 --lr 1e-3 --drop_out 0.7 --gat_atten_drop 0 --is_bn --is_res --out_linear --device 1

# wiki-cs
python node_classification.py --dataset wiki-cs --split public --model GAT --num_layers 2 --hid_size 16 --gat_heads 1 --epochs 2000 --lr 1e-3
python node_classification.py --dataset wiki-cs --split public --model GAT --num_layers 2 --hid_size 512 --gat_heads 4 --epochs 300 --lr 1e-3 --drop_out 0.7 --is_ln --is_res --out_linear --device 1

# GAT for heterophilous datasets

# Amazon-Ratings
python node_classification.py --dataset amazon-ratings --split public --model GAT --num_layers 3 --hid_size 256 --epochs 2000 --lr 1e-2 
python node_classification.py --dataset amazon-ratings --split public --model GAT --hid_size 512 --epochs 2500 --lr 1e-3 --num_layers 4 --weight_decay 0.0 --gat_heads 1 --drop_out 0.5 --is_bn --out_linear --is_res --device 1

# Roman-Empire
python node_classification.py --dataset roman-empire --split public --model GAT --num_layers 2 --hid_size 32 --epochs 2000 --lr 1e-2
python node_classification.py --dataset roman-empire --split public --model GAT --in_linear --num_layers 9 --hid_size 512 --gat_heads 1 --epochs 2000 --lr 1e-3 --drop_out 0.6 --gat_atten_drop 0 --is_bn --is_res --out_linear --device 1

# Minesweeper
python node_classification.py --dataset minesweeper --split public --model GAT --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --drop_out 0.2 --validation_metric auc --device 1
python node_classification.py --dataset minesweeper --split public --model GAT --self_loop --num_layers 12 --hid_size 64 --epochs 800 --lr 1e-2 --drop_out 0.2 --validation_metric auc --is_bn --out_linear --is_res --device 1

# Questions
python node_classification.py --dataset questions --split public --model GAT --num_layers 2 --hid_size 128 --epochs 2000 --lr 1e-2 --drop_out 0.2 --validation_metric auc --device 1
python node_classification.py --dataset questions --split public --model GAT --self_loop --num_layers 9 --hid_size 512 --epochs 3000 --lr 1e-3 --drop_out 0.5 --validation_metric auc --in_linear --is_bn --out_linear --is_res --device 1

# Tolokers
python node_classification.py --dataset tolokers --split public --model GAT --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc
python node_classification.py --dataset tolokers --split public --model GAT --self_loop --num_layers 9 --hid_size 512 --epochs 1500 --lr 1e-3 --validation_metric auc --out_linear --is_bn --is_res --device 1

# GAT for OGB large graph datasets

# ogbn-arxiv (without batch normalization)
python node_classification.py --dataset ogbn-arxiv --split public --model GAT --num_layers 3 --hid_size 128 --num_runs 5 --epochs 800 --lr 1e-2 --use_ogb_eval --device 1
python node_classification.py --dataset ogbn-arxiv --split public --model GAT --self_loop --num_layers 5 --hid_size 512 --num_runs 5 --epochs 2000 --lr 5e-4 --drop_out 0.5 --use_ogb_eval --is_bn --out_linear --is_res --device 0

# ogbn-products (without batch normalization)
python node_classification.py --dataset ogbn-products --split public --model GAT --num_layers 3 --hid_size 256 --num_runs 5 --epochs 600 --lr 1e-2 --drop_out 0.2 --use_ogb_eval
python node_classification.py --dataset ogbn-products --split public --model GAT --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 1000 --lr 3e-3 --drop_out 0.5 --use_ogb_eval --is_ln --out_linear --device 0 

# ogbn-proteins (without batch normalization)
python node_classification.py --dataset ogbn-proteins --split public --model GAT --num_layers 2 --hid_size 16 --num_runs 5 --epochs 300 --lr 1e-2 --loss_fun bce_with_logits --use_ogb_eval --device 0
python node_classification.py --dataset ogbn-proteins --split public --model GAT --self_loop --num_layers 3 --hid_size 512 --num_runs 5 --epochs 100 --lr 1e-2 --drop_out 0.25 --loss_fun bce_with_logits --use_ogb_eval --out_linear --is_res --is_bn

# the others parameters that not explicitly assigned are follow the default configuration.
```

## GraphSAGE

```bash

# GraphSAGE over homophilous datasets

# Cora
python node_classification.py --dataset cora --split public --model SAGE --self_loop --num_layers 2 --hid_size 64 --num_runs 5 --epochs 500 --lr 1e-2

# CiteSeer
python node_classification.py --dataset citeseer --split public --model SAGE --self_loop --num_layers 2 --hid_size 512 --num_runs 5 --epochs 400 --drop_out 0.5 --lr 1e-3

# Pumbed
python node_classification.py --dataset pubmed --split public --model SAGE --self_loop --num_layers 2 --hid_size 256 --num_runs 5 --epochs 400 --drop_out 0.7 --lr 5e-3

# Amazon-Computer
python node_classification.py --dataset am-computer --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 4 --hid_size 512 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset am-computer --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 3 --hid_size 512 --num_runs 5 --epochs 1000 --lr 1e-3 --drop_out 0.5 --is_ln --out_linear --device 1 

# Amazon-Photo
python node_classification.py --dataset am-photo --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 4 --hid_size 128 --num_runs 5 --epochs 1000 --lr 1e-3
python node_classification.py --dataset am-photo --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 6 --hid_size 256 --num_runs 5 --epochs 1000 --lr 1e-3 --drop_out 0.5 --is_ln --is_res --out_linear --device 1

# Coauthor-CS
python node_classification.py --dataset co-cs --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 2 --hid_size 16 --num_runs 5 --epochs 400 --lr 1e-3
python node_classification.py --dataset co-cs --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 2 --hid_size 64 --num_runs 5 --epochs 1500 --lr 1e-3 --drop_out 0.3 --is_ln --is_res --out_linear --device 1

# Coauthor-Physics
python node_classification.py --dataset co-physics --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 2 --hid_size 16 --num_runs 5 --epochs 400 --lr 1e-3
python node_classification.py --dataset co-physics --split random-r-0.6-0.2-0.2 --model SAGE --self_loop --num_layers 2 --hid_size 64 --num_runs 5 --epochs 1500 --lr 1e-3 --drop_out 0.3 --is_ln --is_res --out_linear --device 1

# wiki-cs
python node_classification.py --dataset wiki-cs --split public --model SAGE --self_loop --num_layers 2 --hid_size 16 --epochs 2000 --lr 1e-3
python node_classification.py --dataset wiki-cs --split public --model SAGE --self_loop --num_layers 3 --hid_size 512 --epochs 1500 --lr 1e-3 --drop_out 0.5 --is_ln --out_linear --device 1

# GraphSAGE over heterophilous datasets

# Amazon-Ratings
python node_classification.py --dataset amazon-ratings --split public --model SAGE --self_loop --num_layers 4 --hid_size 128 --epochs 2000 --lr 1e-2
python node_classification.py --dataset amazon-ratings --split public --model SAGE --self_loop --num_layers 4 --hid_size 512 --epochs 600 --lr 1e-3 --drop_out 0.5 --is_bn --out_linear --is_res --device 1

# Roman-Empire (residual connection is crucial)
python node_classification.py --dataset roman-empire --split public --model SAGE --self_loop --num_layers 2 --hid_size 64 --epochs 2000 --lr 1e-2
python node_classification.py --dataset roman-empire --split public --model SAGE --self_loop --num_layers 8 --hid_size 512 --epochs 2500 --lr 1e-3 --drop_out 0.5 --in_linear --out_linear --is_bn --is_res --device 1

# Minesweeper
python node_classification.py --dataset minesweeper --split public --model SAGE --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc 
python node_classification.py --dataset minesweeper --split public --model SAGE --self_loop --num_layers 12 --hid_size 64 --epochs 800 --lr 1e-2 --drop_out 0.2 --validation_metric auc --is_bn --out_linear --is_res --device 1

# Questions
python node_classification.py --dataset questions --split public --model SAGE --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc
python node_classification.py --dataset questions --split public --model SAGE --self_loop --num_layers 9 --hid_size 512 --epochs 1500 --lr 3e-5 --drop_out 0.3 --validation_metric auc --in_linear --out_linear --is_res --device 1

# Tolokers
python node_classification.py --dataset tolokers --split public --model SAGE --self_loop --num_layers 3 --hid_size 64 --epochs 2000 --lr 1e-2 --validation_metric auc
python node_classification.py --dataset tolokers --split public --model SAGE --self_loop --num_layers 9 --hid_size 512 --epochs 1500 --lr 3e-5 --drop_out 0.3 --validation_metric auc --in_linear --out_linear --is_res --device 1

# GraphSAGE over OGB large graphs

# ogbn-arxiv (without batch normalization)
python node_classification.py --dataset ogbn-arxiv --split public --model SAGE --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 800 --lr 1e-2 --use_ogb_eval
python node_classification.py --dataset ogbn-arxiv --split public --model SAGE --self_loop --num_layers 5 --hid_size 512 --num_runs 5 --epochs 2000 --lr 5e-4 --drop_out 0.5 --use_ogb_eval --is_bn --out_linear --is_res --device 1

# ogbn-products (without batch normalization)
python node_classification.py --dataset ogbn-products --split public --model SAGE --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 300 --lr 1e-2 --use_ogb_eval
python node_classification.py --dataset ogbn-products --split public --model SAGE --self_loop --num_layers 3 --hid_size 256 --num_runs 5 --epochs 300 --lr 3e-3 --drop_out 0.5 --use_ogb_eval --is_ln --out_linear --device 0 

# ogbn-proteins (without batch normalization)
python node_classification.py --dataset ogbn-proteins --split public --model SAGE --self_loop --num_layers 2 --hid_size 16 --num_runs 5 --epochs 300 --lr 1e-2 --loss_fun bce_with_logits --use_ogb_eval
python node_classification.py --dataset ogbn-proteins --split public --model SAGE --self_loop --num_layers 3 --hid_size 512 --num_runs 5 --epochs 100 --lr 1e-2 --drop_out 0.25 --loss_fun bce_with_logits --use_ogb_eval --out_linear --is_res --is_bn

# the others parameters that not explicitly assigned are follow the default configuration.
```