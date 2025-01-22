# Node classification performance

Values with '$\downarrow$' indicate that the performance (in our implementation) is (significantly) lower than the
results
reported
in [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://github.com/LUOyk1999/tunedGNN/tree/main).

Values with '$\uparrow$' indicate that the fine-tuned GNN performance is higher than that of standard GNNs.

## Homophilous datasets

|            |              cora               |            citeseer             |             pubmed              |           am-computer           |          am-photo          |           co-cs            |               co-physics                |          wiki-cs           |
|------------|:-------------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|:--------------------------:|:--------------------------:|:---------------------------------------:|:--------------------------:|
| GCN        | $82.56^{\downarrow}_{\pm 0.69}$ | $71.72^{\downarrow}_{\pm 0.26}$ | $79.42^{\downarrow}_{\pm 0.12}$ |       $90.45_{\pm 0.76}$        |     $93.36_{\pm 0.21}$     |     $93.26_{\pm 0.21}$     |           $96.07_{\pm 0.18}$            |     $78.67_{\pm 0.41}$     |
| GCN*       | $82.56^{\downarrow}_{\pm 0.69}$ | $71.72^{\downarrow}_{\pm 0.26}$ | $79.42^{\downarrow}_{\pm 0.12}$ |   $92.10_{\pm 0.36}\uparrow$    | $95.41_{\pm 0.32}\uparrow$ | $95.50_{\pm 0.19}\uparrow$ |       $97.23_{\pm 0.10}\uparrow$        | $79.61_{\pm 0.50}\uparrow$ |
| GAT        | $82.90^{\downarrow}_{\pm 0.58}$ | $71.42^{\downarrow}_{\pm 0.92}$ | $77.92^{\downarrow}_{\pm 0.53}$ |       $89.60_{\pm 0.38}$        |     $92.63_{\pm 0.54}$     |     $90.66_{\pm 0.71}$     |           $95.35_{\pm 0.15}$            |     $74.89_{\pm 0.62}$     |
| GAT*       | $82.90^{\downarrow}_{\pm 0.58}$ | $71.42^{\downarrow}_{\pm 0.92}$ | $77.92^{\downarrow}_{\pm 0.53}$ |   $92.07_{\pm 0.58}\uparrow$    | $95.54_{\pm 0.34}\uparrow$ | $94.98_{\pm 0.10}\uparrow$ |       $96.97_{\pm 0.17}\uparrow$        | $79.74_{\pm 0.47}\uparrow$ |
| GraphSAGE  | $80.32^{\downarrow}_{\pm 0.36}$ | $69.62^{\downarrow}_{\pm 0.56}$ | $77.94^{\downarrow}_{\pm 0.35}$ | $92.31^{\downarrow}_{\pm 0.72}$ |     $95.36_{\pm 0.11}$     |     $94.21_{\pm 0.22}$     |           $96.85_{\pm 0.20}$            |     $79.50_{\pm 0.45}$     |
| GraphSAGE* | $80.32^{\downarrow}_{\pm 0.36}$ | $69.62^{\downarrow}_{\pm 0.56}$ | $77.94^{\downarrow}_{\pm 0.35}$ | $92.04^{\downarrow}_{\pm 0.67}$ | $95.67_{\pm 0.37}\uparrow$ | $95.62_{\pm 0.36}\uparrow$ | $97.03^{\downarrow}_{\pm 0.14}\uparrow$ | $79.88_{\pm 0.65}\uparrow$ |

## Heterophilous datasets

|            |         amazon-ratings          |          roman-empire           |      minesweeper(ROC-AUC)       |           questions(ROC-AUC)            |     tolokers(ROC-AUC)      |
|------------|:-------------------------------:|:-------------------------------:|:-------------------------------:|:---------------------------------------:|:--------------------------:|
| GCN        |       $47.17_{\pm 0.45}$        | $51.86^{\downarrow}_{\pm 0.34}$ | $72.81^{\downarrow}_{\pm 1.26}$ |           $74.49_{\pm 1.11}$            |     $76.56_{\pm 1.11}$     |
| GCN*       |   $53.14_{\pm 0.70}\uparrow$    |   $91.08_{\pm 0.34}\uparrow$    |   $97.32_{\pm 0.32}\uparrow$    |       $77.43_{\pm 1.19}\uparrow$        | $85.11_{\pm 0.54}\uparrow$ |
| GAT        | $46.33^{\downarrow}_{\pm 0.43}$ | $56.78^{\downarrow}_{\pm 0.39}$ |       $87.51_{\pm 1.34}$        |           $70.64_{\pm 1.11}$            |     $79.82_{\pm 1.10}$     |
| GAT*       |   $54.72_{\pm 0.47}\uparrow$    |   $89.95_{\pm 0.32}\uparrow$    |   $96.83_{\pm 0.67}\uparrow$    |       $76.39_{\pm 1.21}\uparrow$        |     $79.82_{\pm 1.10}$     |
| GraphSAGE  | $48.12^{\downarrow}_{\pm 0.68}$ |       $77.45_{\pm 0.51}$        |       $91.82_{\pm 0.53}$        |           $74.73_{\pm 1.06}$            | $83.67_{\pm 0.64}\uparrow$ |
| GraphSAGE* |   $54.40_{\pm 0.45}\uparrow$    |   $89.76_{\pm 0.39}\uparrow$    |   $96.91_{\pm 0.30}\uparrow$    | $74.82^{\downarrow}_{\pm 0.82}\uparrow$ |     $78.03_{\pm 1.13}$     |

## OGB large graphs

|            |         ogbn-arxiv         |       ogbn-products        |     ogbn-proteins(ROC-AUC)      |
|------------|:--------------------------:|:--------------------------:|:-------------------------------:|
| GCN        |     $69.01_{\pm 0.11}$     |     $74.89_{\pm 0.10}$     | $56.16^{\downarrow}_{\pm 0.01}$ |
| GCN*       | $73.27_{\pm 0.24}\uparrow$ | $77.50_{\pm 0.10}\uparrow$ |   $73.33_{\pm 0.43}\uparrow$    |
| GAT        |     $69.24_{\pm 0.11}$     |     $76.66_{\pm 0.88}$     | $50.98^{\downarrow}_{\pm 1.05}$ | 
| GAT*       | $72.87_{\pm 0.17}\uparrow$ |     $78.58_{\pm 0.38}$     |               $$                | 
| GraphSAGE  |             $$             |             $$             |               $$                | 
| GraphSAGE* |             $$             |             $$             |               $$                |