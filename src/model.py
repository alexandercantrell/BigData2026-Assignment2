import torch as ch
from torch_geometric.nn import HeteroConv, SAGEConv

class Model(ch.nn.Module):
    def __init__(self, data,hidden,layers,out,aggr='all'):
        super().__init__()
        aggr = ['mean','max','sum'] if aggr=='all' else aggr
        self.encode = ch.nn.ModuleDict(
            {node_type: ch.nn.Linear(data[node_type].x.size(1),hidden) for node_type in data.node_types}
        )
        self.convs = ch.nn.ModuleList(
            HeteroConv(
                {edge_type: SAGEConv(hidden,hidden,aggr=aggr) for edge_type in data.edge_types},
                aggr='sum')
        for _ in range(layers))
        self.norms = ch.nn.ModuleList(
            ch.nn.ModuleDict({node_type: ch.nn.LayerNorm(hidden) for node_type in data.node_types})
            for _ in range(layers)
        )
        self.head = ch.nn.Linear(hidden,out)

    def forward(self,data,users):
        x = {node_type: self.encode[node_type](data[node_type].x).relu() for node_type in data.node_types}
        for conv, norm in zip (self.convs,self.norms):
            h = conv(x,data.edge_index_dict)
            x = {node_type: (x[node_type] + norm[node_type](h[node_type]).relu()) if node_type in h else x[node_type] for node_type in x}
        return self.head(x["Users"][users])