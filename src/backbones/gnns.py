import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AvgPooling, GATConv, GINConv, SAGEConv, GraphConv, MaxPooling, SumPooling


def readout_mapping(pool: str) -> nn.Module:
    if pool == 'mean':
        return AvgPooling()
    elif pool == 'max':
        return MaxPooling()
    elif pool == 'sum':
        return SumPooling()
    else:
        raise NotImplementedError(pool)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(
            self, in_size, out_size, hid_size=16,
            num_layers=2, activation=F.relu, pooling='mean', drop_out=0,
            in_linear=False, out_linear=False,
            is_res=False, is_ln=False, is_bn=False, is_jk=False
    ):
        super(GIN, self).__init__()
        if in_linear:
            self.linear_input = nn.Linear(in_size, hid_size)
            in_size = hid_size

        self.in_linear = in_linear
        self.out_linear = out_linear
        self.is_res = is_res
        self.is_ln = is_ln
        self.is_bn = is_bn
        self.is_jk = is_jk

        self.res_lins = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.pooling = pooling
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(drop_out)
        self.activation = activation

        if pooling is not None:
            self.readout = readout_mapping(pooling)

        for i in range(num_layers):
            if i == 0:
                mlp = MLP(in_size, hid_size, hid_size)
            else:
                mlp = MLP(hid_size, hid_size, hid_size)

            self.layers.append(
                GINConv(mlp, learn_eps=False)
            )
            self.batch_norms.append(nn.BatchNorm1d(hid_size))

        self.linear_prediction = nn.Linear(hid_size, out_size)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for lin in self.res_lins:
            lin.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        for ln in self.layer_norms:
            ln.reset_parameters()
        if self.in_linear:
            self.linear_input.reset_parameters()
        self.linear_prediction.reset_parameters()

    def forward(self, graph, feats):
        h = feats

        if self.in_linear:
            h = self.linear_input(h)
            h = self.dropout(h)

        h_final = 0

        for i, layer in enumerate(self.layers):
            if self.is_res:
                h = layer(graph, h) + self.res_lins[i](h)
            else:
                h = layer(graph, h)

            if (i != len(self.layers) - 1) or self.out_linear:
                if self.is_ln:
                    h = self.layer_norms[i](h)
                elif self.is_bn:
                    h = self.batch_norms[i](h)
                else:
                    pass

                h = self.activation(h)
                h = self.dropout(h)

            if self.is_jk:
                h_final = h_final + h
            else:
                h_final = h

        if self.pooling is not None:
            if self.out_linear:
                return self.linear_prediction(self.readout(graph, h_final))
            else:
                return self.readout(graph, h_final)
        else:
            if self.out_linear:
                return self.linear_prediction(h_final)
            else:
                return h_final


class GCN(nn.Module):
    def __init__(
            self, in_size, out_size, hid_size=16,
            num_layers=2, activation=F.relu, pooling=None, drop_out=0,
            in_linear=False, out_linear=False,
            is_res=False, is_ln=False, is_bn=False, is_jk=False
    ):
        super(GCN, self).__init__()

        if in_linear:
            self.linear_input = nn.Linear(in_size, hid_size)
            in_size = hid_size

        self.in_linear = in_linear
        self.out_linear = out_linear
        self.is_res = is_res
        self.is_ln = is_ln
        self.is_bn = is_bn
        self.is_jk = is_jk

        self.res_lins = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.pooling = pooling
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(drop_out)
        self.activation = activation

        if pooling is not None:
            self.readout = readout_mapping(pooling)

        for i in range(num_layers):
            if i == 0:
                self.res_lins.append(nn.Linear(in_size, hid_size))
                self.layer_norms.append(nn.LayerNorm(hid_size))
                self.batch_norms.append(nn.BatchNorm1d(hid_size))
                self.layers.append(
                    GraphConv(
                        in_size, hid_size, allow_zero_in_degree=True
                    )
                )
            elif i == num_layers - 1 and not self.out_linear:
                self.layers.append(
                    GraphConv(
                        hid_size, out_size, allow_zero_in_degree=True
                    )
                )
            else:
                self.res_lins.append(nn.Linear(hid_size, hid_size))
                self.layer_norms.append(nn.LayerNorm(hid_size))
                self.batch_norms.append(nn.BatchNorm1d(hid_size))
                self.layers.append(
                    GraphConv(
                        hid_size, hid_size, allow_zero_in_degree=True
                    )
                )

        self.linear_prediction = nn.Linear(hid_size, out_size)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for lin in self.res_lins:
            lin.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        for ln in self.layer_norms:
            ln.reset_parameters()
        if self.in_linear:
            self.linear_input.reset_parameters()
        self.linear_prediction.reset_parameters()

    def forward(self, graph, feats, edge_weight=None):
        h = feats

        if self.in_linear:
            h = self.linear_input(h)
            h = self.dropout(h)

        h_final = 0

        for i, layer in enumerate(self.layers):
            if self.is_res:
                h = layer(graph, h, edge_weight=edge_weight) + self.res_lins[i](h)
            else:
                h = layer(graph, h, edge_weight=edge_weight)

            if (i != len(self.layers) - 1) or self.out_linear:
                if self.is_ln:
                    h = self.layer_norms[i](h)
                elif self.is_bn:
                    h = self.batch_norms[i](h)
                else:
                    pass

                h = self.activation(h)
                h = self.dropout(h)

            if self.is_jk:
                h_final = h_final + h
            else:
                h_final = h

        if self.pooling is not None:
            if self.out_linear:
                return self.linear_prediction(self.readout(graph, h_final))
            else:
                return self.readout(graph, h_final)
        else:
            if self.out_linear:
                return self.linear_prediction(h_final)
            else:
                return h_final


class GAT(nn.Module):
    def __init__(
            self, in_size, out_size, num_heads, hid_size=8,
            num_layers=2, activation=F.relu, pooling=None,
            drop_out=0.6, atten_drop=0.6, in_linear=False, out_linear=False,
            is_res=False, is_ln=False, is_bn=False, is_jk=False
    ):
        super().__init__()

        if in_linear:
            self.linear_input = nn.Linear(in_size, hid_size)
            in_size = hid_size

        self.in_linear = in_linear
        self.out_linear = out_linear
        self.is_res = is_res
        self.is_ln = is_ln
        self.is_bn = is_bn
        self.is_jk = is_jk

        self.res_lins = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.pooling = pooling
        self.batch_norms = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(drop_out)
        self.atten_drop = atten_drop

        if pooling is not None:
            self.readout = readout_mapping(pooling)

        for i in range(num_layers):
            if i == 0:
                self.res_lins.append(nn.Linear(in_size, hid_size))
                self.layer_norms.append(nn.LayerNorm(hid_size))
                self.batch_norms.append(nn.BatchNorm1d(hid_size))
                self.layers.append(
                    GATConv(
                        in_size, hid_size, num_heads, allow_zero_in_degree=True, bias=False, attn_drop=atten_drop
                    )
                )
            elif i == num_layers - 1 and not self.out_linear:
                self.layers.append(
                    GATConv(
                        hid_size, out_size, num_heads, allow_zero_in_degree=True, bias=False, attn_drop=atten_drop
                    )
                )
            else:
                self.res_lins.append(nn.Linear(hid_size, hid_size))
                self.layer_norms.append(nn.LayerNorm(hid_size))
                self.batch_norms.append(nn.BatchNorm1d(hid_size))
                self.layers.append(
                    GATConv(
                        hid_size, hid_size, num_heads, allow_zero_in_degree=True, bias=False, attn_drop=atten_drop
                    )
                )
        self.linear_prediction = nn.Linear(hid_size, out_size)

    def forward(self, graph, feats):
        h = feats

        if self.in_linear:
            h = self.linear_input(h)
            h = self.dropout(h)

        h_final = 0

        for i, layer in enumerate(self.layers):
            if self.is_res:
                h = layer(graph, h).mean(1) + self.res_lins[i](h)
            else:
                h = layer(graph, h).mean(1)

            if (i != len(self.layers) - 1) or self.out_linear:
                if self.is_ln:
                    h = self.layer_norms[i](h)
                elif self.is_bn:
                    h = self.batch_norms[i](h)
                else:
                    pass

                h = self.activation(h)
                h = self.dropout(h)

            if self.is_jk:
                h_final = h_final + h
            else:
                h_final = h

        if self.pooling is not None:
            if self.out_linear:
                return self.linear_prediction(self.readout(graph, h_final))
            else:
                return self.readout(graph, h_final)
        else:
            if self.out_linear:
                return self.linear_prediction(h_final)
            else:
                return h_final


class SAGE(nn.Module):
    def __init__(
            self, in_size, out_size, hid_size=16,
            num_layers=2, activation=F.relu, pooling=None, drop_out=0,
            in_linear=False, out_linear=False,
            is_res=False, is_ln=False, is_bn=False, is_jk=False
    ):
        super(SAGE, self).__init__()

        if in_linear:
            self.linear_input = nn.Linear(in_size, hid_size)
            in_size = hid_size

        self.in_linear = in_linear
        self.out_linear = out_linear
        self.is_res = is_res
        self.is_ln = is_ln
        self.is_bn = is_bn
        self.is_jk = is_jk

        self.res_lins = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layers = nn.ModuleList()

        self.pooling = pooling
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(drop_out)
        self.activation = activation

        if pooling is not None:
            self.readout = readout_mapping(pooling)

        for i in range(num_layers):
            if i == 0:
                self.res_lins.append(nn.Linear(in_size, hid_size))
                self.layer_norms.append(nn.LayerNorm(hid_size))
                self.batch_norms.append(nn.BatchNorm1d(hid_size))
                self.layers.append(
                    SAGEConv(
                        in_size, hid_size, aggregator_type='mean'
                    )
                )
            elif i == num_layers - 1 and not self.out_linear:
                self.layers.append(
                    SAGEConv(
                        hid_size, out_size, aggregator_type='mean'
                    )
                )
            else:
                self.res_lins.append(nn.Linear(hid_size, hid_size))
                self.layer_norms.append(nn.LayerNorm(hid_size))
                self.batch_norms.append(nn.BatchNorm1d(hid_size))
                self.layers.append(
                    SAGEConv(
                        hid_size, hid_size, aggregator_type='mean'
                    )
                )

        self.linear_prediction = nn.Linear(hid_size, out_size)

    def forward(self, graph, feats, edge_weight=None):
        h = feats

        if self.in_linear:
            h = self.linear_input(h)
            h = self.dropout(h)

        h_final = 0

        for i, layer in enumerate(self.layers):
            if self.is_res:
                h = layer(graph, h, edge_weight=edge_weight) + self.res_lins[i](h)
            else:
                h = layer(graph, h, edge_weight=edge_weight)

            if (i != len(self.layers) - 1) or self.out_linear:
                if self.is_ln:
                    h = self.layer_norms[i](h)
                elif self.is_bn:
                    h = self.batch_norms[i](h)
                else:
                    pass

                h = self.activation(h)
                h = self.dropout(h)

            if self.is_jk:
                h_final = h_final + h
            else:
                h_final = h

        if self.pooling is not None:
            if self.out_linear:
                return self.linear_prediction(self.readout(graph, h_final))
            else:
                return self.readout(graph, h_final)
        else:
            if self.out_linear:
                return self.linear_prediction(h_final)
            else:
                return h_final
