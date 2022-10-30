import os
from time import time
from turtle import forward
import torch
import torch.nn.functional as F
from .graph_layer import GraphConvolution
import torch.nn as nn
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.norm import GraphNorm

from transformers import RobertaTokenizerFast, RobertaModel, BertTokenizer, BertModel

class Attblock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, trg, src_mask):
        #q = k = self.with_pos_embed(src, pos)
        q = src.permute(1,0,2)
        k = trg.permute(1,0,2)
        src_mask = ~src_mask.bool()
        # please see https://zhuanlan.zhihu.com/p/353365423 for the funtion of key_padding_mask
        src2, attention = self.attn(q, k, value=k, key_padding_mask=src_mask)
        src2 = src2.permute(1,0,2)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attention

class GATv2(nn.Module):
    def __init__(self, input_dim, output_dim, graph_norm=nn.Identity, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        # no need to add_self_loops, since this is done in Perception.forward()
        
        self.gat1 = GATv2Conv(input_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gn1 = graph_norm(in_channels=hidden_dim)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gn2 = graph_norm(in_channels=hidden_dim)
        self.gat3 = GATv2Conv(input_dim, output_dim, dropout=dropout, add_self_loops=False)
        self.gn3 = graph_norm(in_channels=output_dim)

    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]

        # NOTE: Three ways of producing edge_indices were tested:

        # (i) create a large all-zero tensor and copy each adj mat onto it. This is implemented as the following 4 lines
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        edge_indices = torch.nonzero(big_adj > 0).t()

        # (ii) [This method is used now] used torch.block_diag to compose all adj mats. it is implemented using "edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()"
        # adj_list = [adj[b] for b in range(B)]
        # edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        # (iii) extract edge indices of each adj mat and then concatenate them. it is implemented using "edge_indices = torch.cat([torch.nonzero(adj_batch[b] > 0).t() + b*N for b in range(B)], dim=1)"
        # the running time of the three ways are compared: (ii) < (i) < (iii), which means (ii) is the fastest
        
        # NOTE: GATv2Conv requires edge indices as input
        # NOTE: ordering matters. -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout
        x = self.gn1(self.gat1(big_graph, edge_indices))
        x = self.gn2(self.gat2(x, edge_indices))
        big_output = self.gn3(self.gat3(x, edge_indices, return_attention_weights=True if return_attention_weights else None))

        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            # NOTE: att_scores has a form of [a11, a21, ..., an1 | a12, a22, ..., an2, |.... | a1n, a2n, ... ann]
            batch_output, edge_indices_and_att_scores = torch.stack(big_output[0].split(N)), big_output[1]
            raw_att_scores = edge_indices_and_att_scores[1][:,0]

            degree_mat = adj.sum(dim=2).int() # B x N
            att_scores = []
            for b in range(B):
                adj_mat = degree_mat[b] # this adj matrix contains the global node and self-loops while that in obs does not
                idxs = [0]
                for i in range(adj_mat.shape[0] - 1):
                    idxs.append(idxs[-1] + adj_mat[i].item())
                att_scores.append(raw_att_scores[idxs])
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores

class Custom_GATv2(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        # no need to add_self_loops, since this is done in Perception.forward()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GATv2Conv(input_dim, hidden_dim, add_self_loops=False))
            elif i == num_layers-1:
                layers.append(GATv2Conv(hidden_dim, output_dim, add_self_loops=False))
            else:
                layers.append(GATv2Conv(hidden_dim, hidden_dim, add_self_loops=False))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]

        # NOTE: Three ways of producing edge_indices were tested:

        # (i) create a large all-zero tensor and copy each adj mat onto it. This is implemented as the following 4 lines
        # big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        # for b in range(B):
        #     big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        # edge_indices = torch.nonzero(big_adj > 0).t()

        # (ii) [This method is used now] used torch.block_diag to compose all adj mats. it is implemented using "edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()"
        adj_list = [adj[b] for b in range(B)]
        edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        # (iii) extract edge indices of each adj mat and then concatenate them. it is implemented using "edge_indices = torch.cat([torch.nonzero(adj_batch[b] > 0).t() + b*N for b in range(B)], dim=1)"
        # the running time of the three ways are compared: (ii) < (i) < (iii), which means (ii) is the fastest
        
        # GATv2Conv requires edge indices as input
        for i in range(len(self.layers) - 1):
            big_graph =self.layers[i](big_graph,edge_indices)

        big_output = self.layers[-1](big_graph,edge_indices, return_attention_weights=True if return_attention_weights else None)
        
        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            batch_output, att_scores = torch.stack(big_output[0].split(N)), big_output[1]
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores
    
class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        self.gat1 = GATConv(input_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, dropout=dropout, add_self_loops=False)
        self.gat3 = GATConv(input_dim, output_dim, add_self_loops=False)
    
    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        # big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        # for b in range(B):
        #     big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        # # GATv2Conv requires edge indices as input
        # edge_indices = torch.nonzero(big_adj > 0).t()
        
        # This is fatser
        adj_list = [adj[b] for b in range(B)]
        edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        x = self.gat1(big_graph, edge_indices)
        x = self.gat2(x, edge_indices)
        big_output = self.gat3(x, edge_indices, return_attention_weights=True if return_attention_weights else None)

        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            batch_output, att_scores = torch.stack(big_output[0].split(N)), big_output[1]
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores

class Custom_GAT(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, dropout=0.1, hidden_dim=512) -> None:
        super().__init__()
        # leaky_ReLU and dropout are built in GATv2Conv class
        # no need to add_self_loops, since this is done in Perception.forward()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GATConv(input_dim, hidden_dim, dropout=dropout, add_self_loops=False))
            elif i == num_layers-1:
                layers.append(GATConv(hidden_dim, output_dim, dropout=dropout, add_self_loops=False))
            else:
                layers.append(GATConv(hidden_dim, hidden_dim, add_self_loops=False))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]

        # NOTE: Three ways of producing edge_indices were tested:

        # (i) create a large all-zero tensor and copy each adj mat onto it. This is implemented as the following 4 lines
        # big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        # for b in range(B):
        #     big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]
        # edge_indices = torch.nonzero(big_adj > 0).t()

        # (ii) [This method is used now] used torch.block_diag to compose all adj mats. it is implemented using "edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()"
        adj_list = [adj[b] for b in range(B)]
        edge_indices = torch.nonzero(torch.block_diag(*adj_list) > 0).t()

        # (iii) extract edge indices of each adj mat and then concatenate them. it is implemented using "edge_indices = torch.cat([torch.nonzero(adj_batch[b] > 0).t() + b*N for b in range(B)], dim=1)"
        # the running time of the three ways are compared: (ii) < (i) < (iii), which means (ii) is the fastest
        
        # GATv2Conv requires edge indices as input
        for i in range(len(self.layers) - 1):
            big_graph =self.layers[i](big_graph,edge_indices)

        big_output = self.layers[-1](big_graph,edge_indices, return_attention_weights=True if return_attention_weights else None)
        
        att_scores = None
        if return_attention_weights: # if we need attention scores of GATv2
            batch_output, att_scores = torch.stack(big_output[0].split(N)), big_output[1]
        else:
            batch_output = torch.stack(big_output.split(N))

        return batch_output, att_scores
    
class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout =0.1, hidden_dim=512, init='xavier'):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim, init=init)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim, init=init)
        self.gc3 = GraphConvolution(hidden_dim, output_dim, init=init)
        self.dropout = nn.Dropout(dropout)

    def normalize_sparse_adj(self, adj):
        """Laplacian Normalization"""
        rowsum = adj.sum(1) # adj B * M * M
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        
        r_inv_sqrt[torch.where(torch.isinf(r_inv_sqrt))] = 0.
        r_mat_inv_sqrt = torch.stack([torch.diag(k) for k in r_inv_sqrt])
        
        return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(1,2),r_mat_inv_sqrt)


    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        adj = self.normalize_sparse_adj(adj)
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        x = self.dropout(F.relu(self.gc1(big_graph,big_adj)))
        x = self.dropout(F.relu(self.gc2(x,big_adj)))
        
        big_output = self.gc3(x, big_adj)

        big_adj[:] = 0.
        x = self.dropout(F.relu(self.gc1(big_graph,big_adj)))
        x = self.dropout(F.relu(self.gc2(x,big_adj)))
        
        big_output = self.gc3(x, big_adj)

        batch_output = torch.stack(big_output.split(N))
        return batch_output

class Custom_GCN(nn.Module): 
    def __init__(self, input_dim, output_dim, num_layers=3, dropout =0.1, hidden_dim=512, init='xavier'):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GraphConvolution(input_dim, hidden_dim, init=init))
            elif i == num_layers-1:
                layers.append(GraphConvolution(hidden_dim, output_dim, init=init))
            else:
                layers.append(GraphConvolution(hidden_dim, hidden_dim, init=init))

        self.layers = nn.ModuleList(layers)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def normalize_sparse_adj(self, adj):
        """Laplacian Normalization"""
        rowsum = adj.sum(1) # adj B * M * M
        r_inv_sqrt = torch.pow(rowsum, -0.5)
        r_inv_sqrt[torch.where(torch.isinf(r_inv_sqrt))] = 0.
        r_mat_inv_sqrt = torch.stack([torch.diag(k) for k in r_inv_sqrt])
        return torch.matmul(torch.matmul(adj, r_mat_inv_sqrt).transpose(1,2),r_mat_inv_sqrt)

    def forward(self, batch_graph, adj, return_attention_weights=False):
        # make the batch into one graph
        big_graph = torch.cat([graph for graph in batch_graph],0)
        B, N = batch_graph.shape[0], batch_graph.shape[1]
        big_adj = torch.zeros(B*N, B*N).to(batch_graph.device)
        for b in range(B):
            big_adj[b*N:(b+1)*N,b*N:(b+1)*N] = adj[b]

        for i in range(len(self.layers)):
            if i != len(self.layers) - 1:
                big_graph = self.dropout(F.relu(self.layers[i](big_graph,big_adj)))
            else:
                big_graph = self.layers[i](big_graph,big_adj)

        batch_output = torch.stack(big_graph.split(N))
        return batch_output

import math
class PositionEncoding(nn.Module):
    def __init__(self, n_filters=512, max_len=2000):
        """
        :param n_filters: same with input hidden size
        :param max_len: maximum sequence length
        """
        super(PositionEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_filters)  # (L, D)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_filters, 2).float() * - (math.log(10000.0) / n_filters))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # buffer is a tensor, not a variable, 510 x 512

    def forward(self, x, times):
        """
        x: B x num_nodes x 512
        times: B x num_nodes
        """
        
        pe = []
        for b in range(x.shape[0]):
            pe.append(self.pe.data[times[b].long()]) # (#x.size(-2), n_filters)
        pe_tensor = torch.stack(pe) # B x num_nodes x 512
        x = x + pe_tensor
        return x

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 800):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1) # seq_len x 1 x 512
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class Perception(nn.Module):
    def __init__(self,config):
        super(Perception, self).__init__()
        self.pe_method = 'pe' # or exp(-t)
        self.time_embedd_size = config.features.time_dim
        self.max_time_steps = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        self.goal_time_embedd_index = self.max_time_steps
        memory_dim = config.features.visual_feature_dim
        self.memory_dim = memory_dim

        if self.pe_method == 'embedding':
            self.time_embedding = nn.Embedding(self.max_time_steps+2, self.time_embedd_size)
        elif self.pe_method == 'pe':
            self.time_embedding = PositionEncoding(memory_dim, self.max_time_steps+10)
        else:
            self.time_embedding = lambda t: torch.exp(-t.unsqueeze(-1)/5)

        feature_dim = config.features.visual_feature_dim# + self.time_embedd_size
        #self.feature_embedding = nn.Linear(feature_dim, memory_dim)
        self.feature_embedding = nn.Sequential(nn.Linear(feature_dim +  config.features.visual_feature_dim , memory_dim),
                                               nn.ReLU(),
                                               nn.Linear(memory_dim, memory_dim))
        
        gn_dict = {
            "graph_norm": GraphNorm
        }
        gn = gn_dict.get(config.GCN.GRAPH_NORM, nn.Identity)

        if config.GCN.TYPE == "GCN":
            self.global_GCN = GCN(input_dim=memory_dim, output_dim=memory_dim)
        elif config.GCN.TYPE == "GAT":
            self.global_GCN = GAT(input_dim=memory_dim, output_dim=memory_dim)
        elif config.GCN.TYPE == "GATv2":
            self.global_GCN = GATv2(input_dim=memory_dim, output_dim=memory_dim, graph_norm=gn)
        
        # if config.GCN.WITH_ENV_GLOBAL_NODE:
        #     self.with_env_global_node = True
        #     self.env_global_node_respawn = config.GCN.RESPAWN_GLOBAL_NODE
        #     self.randominit_env_global_node = config.GCN.RANDOMINIT_ENV_GLOBAL_NODE
        #     node_vec = torch.randn(1, memory_dim) if self.randominit_env_global_node else torch.zeros(1, memory_dim)
        #     self.env_global_node = torch.nn.parameter.Parameter(node_vec, requires_grad=False)

        #     #self.env_global_node_each_proc = self.env_global_node.unsqueeze(0).repeat(config.NUM_PROCESSES, 1, 1) # it is a torch.Tensor, not Parameter
        # else:
        #     self.with_env_global_node = False
        
        self.goal_Decoder = Attblock(config.transformer.hidden_dim,
                                    config.transformer.nheads, # default to 4
                                    config.transformer.dim_feedforward,
                                    config.transformer.dropout)
        self.curr_Decoder = Attblock(config.transformer.hidden_dim,
                                    config.transformer.nheads,
                                    config.transformer.dim_feedforward,
                                    config.transformer.dropout)

        self.output_size = feature_dim

        # BERT
        self.with_unified_embedding = False
        if config.transformer.with_unified_embedding:
            os.environ['TOKENIZERS_PARALLELISM'] = "true"
            self.with_unified_embedding = True

            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            encoder = BertModel.from_pretrained("bert-base-multilingual-cased")

            self.reduce = nn.Linear(768, config.transformer.hidden_dim)
            with torch.no_grad():
                inputs = tokenizer('Go to ', return_tensors="pt")
                outputs = encoder(**inputs)
                self.imagenav_instruc_emb = outputs.last_hidden_state # 1 x 5 x 768

            self.task_embedding = torch.nn.Embedding(num_embeddings=1, embedding_dim=config.transformer.hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                config.transformer.hidden_dim,
                config.transformer.nheads,
                config.transformer.dim_feedforward,
                dropout=config.transformer.dropout,
                activation='gelu')
            
            self.goal_encoder = nn.TransformerEncoder(
                encoder_layer,
                2,
                norm=nn.LayerNorm(config.transformer.hidden_dim))
            
            self.instruc_posemb = LearnedPositionEncoding(
                d_model=config.transformer.hidden_dim,
                dropout=config.transformer.dropout)
        # if config.transformer.with_tesk_query:
        #     self.image_nav_query = nn.Embedding(1,config.transformer.hidden_dim)
        

    def forward(self, observations, env_global_node, return_features=False, disable_forgetting=False): # without memory
        # env_global_node: b x 1 x 512
        # forgetting mechanism is enabled only when collecting trajectories and it is disabled when evaluating actions
        B = observations['global_mask'].shape[0]
        max_node_num = observations['global_mask'].sum(dim=1).max().long() # this indicates that the elements in global_mask denotes the existence of nodes

        # observations['global_time']: num_process D vector, it contains the timestamps of each node in each navigation process. it is from self.graph_time in graph.py
        # observations['step']: num_process x max_num_node, it is controlled by the for-loop at line 68 in bc_trainer.py, recording the current simulation timestep
        relative_time = observations['step'].unsqueeze(1) - observations['global_time'][:, :max_node_num]

        global_memory = self.time_embedding(observations['global_memory'][:,:max_node_num], relative_time)

        # NOTE: please clone the global mask, because the forgetting mechanism will alter the contents in the original mask and cause undesirable errors (e.g. RuntimeError: CUDA error: device-side assert triggered)
        global_mask = observations['global_mask'][:,:max_node_num].clone() # B x max_num_node. an element is 1 if the node exists
        device = global_memory.device
        I = torch.eye(max_node_num, device=device).unsqueeze(0).repeat(B,1,1)
        global_A = observations['global_A'][:,:max_node_num, :max_node_num]  + I
        goal_embedding = observations['goal_embedding']

        # concatenate graph node features with the target image embedding, which are both 512d vectors,
        # and then project the concatenated features to 512d new features
        # B x max_num_nodes x 512
        global_memory_with_goal = self.feature_embedding(torch.cat((global_memory[:,:max_node_num], goal_embedding.unsqueeze(1).repeat(1,max_node_num,1)),-1))

        # goal_attn: B x output_seq_len (1) x input_seq_len (num_nodes). NOTE: the att maps of all heads are averaged

        #t1 = time()
        if env_global_node is not None: # global node: this block takes 0.0002s
            batch_size, A_dtype = global_A.shape[0], global_A.dtype

            global_memory_with_goal = torch.cat([env_global_node, global_memory_with_goal], dim=1)

            if self.link_fraction != -1:
                add_row = torch.zeros(1, 1, max_node_num, dtype=A_dtype, device=device)
                link_number = max(1, int(self.link_fraction * max_node_num)) # round up
                #link_number = int(self.link_fraction)
                add_row[0,0,-link_number:] = 1.0

                add_col = torch.zeros(1, max_node_num + 1, 1, dtype=A_dtype, device=device)
                add_col[0,-link_number:,0] = 1.0
                add_col[0,0,0] = 1.0

                global_A = torch.cat([add_row, global_A], dim=1)
                global_A = torch.cat([add_col, global_A], dim=2)
                
            else:
                global_A = torch.cat([torch.ones(batch_size, 1, max_node_num, dtype=A_dtype, device=device), global_A], dim=1)
                global_A = torch.cat([torch.ones(batch_size, max_node_num + 1, 1, dtype=A_dtype, device=device), global_A], dim=2)
            

            global_mask = torch.cat([torch.ones(batch_size, 1, dtype=global_mask.dtype, device=device), global_mask], dim=1) # B x (max_num_node+1)

        #print("Preparation time {:.4f}s".format(time()- t1))

        #t1 = time()
        # Speed Profile:
        # GATv2-env_global_node: forward takes 0.0027s at least and 0.1806s at most
        # GATv2: 0.0025s at least and 0.0163s at most
        # GCN: takes 0.0006s at least and 0.0011s at most
        GCN_results = self.global_GCN(global_memory_with_goal, global_A, return_features) # 4 1 512
        
        #print("GCN forward time {:.4f}s".format(time()- t1))
        # GAT_attn is a tuple: (edge_index, alpha)
        GAT_attn = None
        if isinstance(GCN_results, tuple):
            global_context, GAT_attn = GCN_results
        else:
            global_context = GCN_results

        curr_embedding, goal_embedding = observations['curr_embedding'], observations['goal_embedding']

        new_env_global_node = None

        #t1 = time()
        # embedding takes 0.0003s
        if env_global_node is not None: 
            if self.random_replace:
                random_idx = torch.randint(low=1, high=max_node_num+1, size=(1,))
                new_env_global_node = global_context[:,random_idx:random_idx+1]
            else:
                new_env_global_node = global_context[:,0:1] # save the global node features for next time's use

            # the global node along with all nodes act as keys and values
            if self.decode_global_node:
                global_context = torch.cat([new_env_global_node, self.time_embedding(global_context[:,1:], relative_time)], dim=1)
            else:
                global_context = self.time_embedding(global_context[:,1:], relative_time)
        else:
            global_context = self.time_embedding(global_context, relative_time)
        #print("embedding time {:.4f}s".format(time()- t1))
        # global_context = self.time_embedding(global_context, relative_time)
        
        #t1 = time()
        # the two decoding processes take 0.0018s at least and 0.0037 at most
        
        if self.with_unified_embedding:
            instruc_emb = self.reduce(self.imagenav_instruc_emb.to(device)).repeat(B,1,1) # B x seq_len x 512
            goal_embedding = torch.cat([
                self.task_embedding.weight[0].view(1,1,-1).repeat(B,1,1).to(device),
                instruc_emb,
                goal_embedding.unsqueeze(1)
                ], dim=1) # B x (seq_len+1) x 512
            goal_embedding = self.goal_encoder(self.instruc_posemb(goal_embedding.permute(1,0,2))).permute(1,0,2)
        else:
            goal_embedding = goal_embedding.unsqueeze(1)
        goal_context, goal_attn = self.goal_Decoder(goal_embedding[:,0:1,:], global_context, global_mask)
        #print(global_context[0].shape, global_mask[0], goal_attn[0], );input()
        curr_context, curr_attn = self.curr_Decoder(curr_embedding.unsqueeze(1), global_context, global_mask)
        #print("decoder time {:.4f}s".format(time()- t1))


        # print(new_env_global_node[0:2,0,0:10])
        return curr_context.squeeze(1), goal_context.squeeze(1), new_env_global_node, \
            {'goal_attn': goal_attn.squeeze(1),
            'curr_attn': curr_attn.squeeze(1),
            'GAT_attn': GAT_attn if GAT_attn is not None else None,
            'Adj_mat': global_A} if return_features else None
