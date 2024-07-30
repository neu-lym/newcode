import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from utils_package.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
import json

class MENTOR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MENTOR, self).__init__(config, dataset)
        self.sparse = True
        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.embedding_dim = 64
        self.n_ui_layers = config['n_ui_layers']
        self.um_loss = config['um_loss']
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.mask_weight_g = config['mask_weight_g']
        self.mask_weight_f = config['mask_weight_f']
        self.temp = config['temp']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.mm_adj = None
        self.um_temp = config['um_temp']

        self.mlp = nn.Linear(2*dim_x, 2*dim_x)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.data_name = config['dataset']

        self.user_graph_dict = np.load(os.path.join(self.dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        #初始化扩展用户物品图上的用户图像嵌入
        self.extended_image_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_image_user.weight)
        #初始化扩展用户物品图上的用户文本嵌入
        self.extended_text_user = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.extended_text_user.weight)

        self.mlp2 = nn.Linear(64,128)
        self.mlp4 = nn.Linear(128,128)
        self.mlp5 = nn.Linear(64,128)
        #图像和文本的项目项目矩阵路径
        image_adj_file = os.path.join(self.dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(self.dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        mm_adj_file = os.path.join(self.dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        #获得物品德图像嵌入和文本嵌入
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        
         #  Enhancing User-Item Graph
        self.inter = self.find_inter(self.image_original_adj, self.text_original_adj)
        self.ii_adj = self.add_edge(self.inter)
        self.norm_adj = self.get_adj_mat(self.ii_adj.tolil())
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        self.image_reduce_dim = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        self.image_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.image_space_trans = nn.Sequential(
            self.image_reduce_dim,
            self.image_trans_dim
        )
        self.text_reduce_dim = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.text_trans_dim = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_space_trans = nn.Sequential(
            self.text_reduce_dim,
            self.text_trans_dim
        )
        self.separate_coarse = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=-1)
                
        self.image_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.text_behavior = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.weight_t = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_t.data = F.softmax(self.weight_t, dim=1)

        self.weight_i = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent * len(drop_item))]

        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1 / 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)

        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        if self.v_feat is not None:
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
            self.v_gcn_n1 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.v_feat)
            self.v_gcn_n2 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)
            self.t_gcn_n1 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                             device=self.device, features=self.t_feat)
            self.t_gcn_n2 = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode, dim_latent=64,
                                device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))
        self.id_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)


        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

        self.result_embed_guide = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_v = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_t = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_n1 = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
        self.result_embed_n2 = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)

    def find_inter(self, image_adj, text_adj):
        inter_file = os.path.join(self.dataset_path, 'inter.json')
        if os.path.exists(inter_file):
            with open(inter_file) as f:
                inter = json.load(f)
        else:
            j = 0
            inter = defaultdict(list)
            img_sim = []
            txt_sim = []
            for i in range(0,len(image_adj._indices()[0])):
                img_id = image_adj._indices()[0][i]
                txt_id = text_adj._indices()[0][i]
                assert img_id == txt_id
                id = img_id.item()
                img_sim.append(image_adj._indices()[1][j].item())
                txt_sim.append(text_adj._indices()[1][j].item())
                
                if len(img_sim)==10 and len(txt_sim)==10:
                    it_inter = list(set(img_sim) & set(txt_sim))
                    inter[id] = [v for v in it_inter if v != id]
                    img_sim = []
                    txt_sim = []
                
                j += 1
            
            with open(inter_file, "w") as f:
                json.dump(inter, f)
        
        return inter

    def add_edge(self, inter):
        sim_rows = []
        sim_cols = []
        for id, vs in inter.items():
            if len(vs) == 0:
                continue
            for v in vs:
                sim_rows.append(int(id))
                sim_cols.append(v)
        
        sim_rows = torch.tensor(sim_rows)
        sim_cols = torch.tensor(sim_cols)
        sim_values = [1]*len(sim_rows)

        item_adj = sp.coo_matrix((sim_values, (sim_rows, sim_cols)), shape=(self.n_items,self.n_items), dtype=np.int64)
        return item_adj
    def get_adj_mat(self, item_adj):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()

        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        adj_mat[self.n_users:, self.n_users:] = item_adj
        
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        
        return norm_adj_mat.tocsr()
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def conv_ui(self, adj, user_embeds, item_embeds):
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        
        return all_embeddings

    def conv_ii(self, ii_adj, single_modal):
        for i in range(self.n_layers):
            single_modal = torch.sparse.mm(ii_adj, single_modal)
        return single_modal

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        pass
    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)


    def forward(self, interaction,adj,train=False):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.image_space_trans(self.v_feat))#[,64]
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.text_space_trans(self.t_feat))#[,64]

        # GCN for id, v, t modalities
        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index_dropt, self.edge_index, self.id_feat)

        #得到扩展图的id
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        extended_id_embeds = self.conv_ui(adj, user_embeds, item_embeds)
        
        explicit_image_item = self.conv_ii(self.image_original_adj, image_item_embeds)
        explicit_image_user = torch.sparse.mm(self.R, explicit_image_item)
        explicit_image_embeds = torch.cat([explicit_image_user, explicit_image_item], dim=0)

        explicit_text_item = self.conv_ii(self.text_original_adj, text_item_embeds)
        explicit_text_user = torch.sparse.mm(self.R, explicit_text_item)
        explicit_text_embeds = torch.cat([explicit_text_user, explicit_text_item], dim=0)
        
        #得到扩展的图像和文本嵌入
        extended_image_embeds = self.conv_ui(adj, self.extended_image_user.weight, explicit_image_item) 
        extended_text_embeds = self.conv_ui(adj, self.extended_text_user.weight, explicit_text_item)
        extended_it_embeds = (extended_image_embeds + extended_text_embeds) / 2
              
        fine_grained_image =  (explicit_image_embeds + extended_id_embeds) 
        fine_grained_text = (explicit_text_embeds + extended_id_embeds) 
        final_features = (fine_grained_image + fine_grained_text) 
       
        # random noise GCN for v and t
        self.v_rep_n1, _ = self.v_gcn_n1(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n1, _ = self.t_gcn_n1(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)
        self.v_rep_n2, _ = self.v_gcn_n2(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n2, _ = self.t_gcn_n2(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)

        # v, v, id, and vt modalities
        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)

        # noise rep
        representation_n1 = torch.cat((self.v_rep_n1, self.t_rep_n1), dim=1)
        representation_n2 = torch.cat((self.v_rep_n2, self.t_rep_n2), dim=1)

        self.v_rep = torch.unsqueeze(self.v_rep, 2)
        self.t_rep = torch.unsqueeze(self.t_rep, 2)
        self.id_rep = torch.unsqueeze(self.id_rep, 2)

        user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=2)
        guide_user_rep = torch.cat((guide_user_rep[:, :, 0], guide_user_rep[:, :, 1]), dim=1)

        v_user_rep = torch.cat((self.v_rep[:self.num_user], self.v_rep[:self.num_user]), dim=2)
        v_user_rep = torch.cat((v_user_rep[:, :, 0], v_user_rep[:, :, 1]), dim=1)

        t_user_rep = torch.cat((self.t_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
        t_user_rep = torch.cat((t_user_rep[:, :, 0], t_user_rep[:, :, 1]), dim=1)

        # noise rep1
        self.v_rep_n1 = torch.unsqueeze(self.v_rep_n1, 2)
        self.t_rep_n1 = torch.unsqueeze(self.t_rep_n1, 2)
        user_rep_n1 = torch.cat((self.v_rep_n1[:self.num_user], self.t_rep_n1[:self.num_user]), dim=2)
        user_rep_n1 = self.weight_u.transpose(1, 2) * user_rep_n1
        user_rep_n1 = torch.cat((user_rep_n1[:, :, 0], user_rep_n1[:, :, 1]), dim=1)

        # noise rep2
        self.v_rep_n2 = torch.unsqueeze(self.v_rep_n2, 2)
        self.t_rep_n2 = torch.unsqueeze(self.t_rep_n2, 2)
        user_rep_n2 = torch.cat((self.v_rep_n2[:self.num_user], self.t_rep_n2[:self.num_user]), dim=2)
        user_rep_n2 = self.weight_u.transpose(1, 2) * user_rep_n2
        user_rep_n2 = torch.cat((user_rep_n2[:, :, 0], user_rep_n2[:, :, 1]), dim=1)

        item_rep = representation[self.num_user:]
        item_rep_n1 = representation_n1[self.num_user:]
        item_rep_n2 = representation_n2[self.num_user:]

        guide_item_rep = guide_representation[self.num_user:]
        v_item_rep = v_representation[self.num_user:]
        t_item_rep = t_representation[self.num_user:]

        # build item-item graph
        h = self.buildItemGraph(item_rep)
        h_guide = self.buildItemGraph(guide_item_rep)
        h_v = self.buildItemGraph(v_item_rep)
        h_t = self.buildItemGraph(t_item_rep)
        h_n1 = self.buildItemGraph(item_rep_n1)
        h_n2 = self.buildItemGraph(item_rep_n2)

        user_rep = user_rep
        item_rep = item_rep + h
        
        user_rep = self.max_pool(user_rep)
        item_rep = self.max_pool(item_rep)
       
        item_rep_n1 = item_rep_n1 + h_n1
        item_rep_n2 = item_rep_n2 + h_n2

        guide_item_rep = guide_item_rep + h_guide
        v_item_rep = v_item_rep + h_v
        t_item_rep = t_item_rep+ h_t

        # build result embedding
        self.user_rep = user_rep
        self.user_rep = torch.cat((self.user_rep,final_features[:self.n_users]),dim=1)
        self.item_rep = item_rep
        self.item_rep = torch.cat((self.item_rep,final_features[self.n_users:]),dim=1)
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        
        self.guide_user_rep = guide_user_rep
        self.guide_item_rep = guide_item_rep
        self.result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)

        self.v_user_rep = v_user_rep
        self.v_item_rep = v_item_rep
        self.result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)

        self.t_user_rep = t_user_rep
        self.t_item_rep = t_item_rep
        self.result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)

        self.user_rep_n1 = user_rep_n1
        self.item_rep_n1 = item_rep_n1
        self.result_embed_n1 = torch.cat((user_rep_n1, item_rep_n1), dim=0)

        self.user_rep_n2 = user_rep_n2
        self.item_rep_n2 = item_rep_n2
        self.result_embed_n2 = torch.cat((user_rep_n2, item_rep_n2), dim=0)

        # calculate pos and neg scores
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores
    def buildItemGraph(self, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def fit_Gaussian_dis(self):
        r_var = torch.var(self.result_embed)
        r_mean = torch.mean(self.result_embed)
        g_var = torch.var(self.result_embed_guide)
        g_mean = torch.mean(self.result_embed_guide)
        v_var = torch.var(self.result_embed_v)
        v_mean = torch.mean(self.result_embed_v)
        t_var = torch.var(self.result_embed_t)
        t_mean = torch.mean(self.result_embed_t)
        return r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean

    def cal_noise_loss(self, id, emb, temp):

        def add_perturbation(x):
            random_noise = torch.rand_like(x).to(self.device)
            x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
            return x

        emb_view1 = add_perturbation(emb)
        emb_view2 = add_perturbation(emb)
        emb_loss = self.InfoNCE(emb_view1[id], emb_view2[id], temp)

        return emb_loss
    
    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction,self.norm_adj,train=True)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
       
        # reg
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        # mask
        # with torch.no_grad():
        #     u_temp, i_temp = self.user_rep.clone(), self.item_rep.clone()
        #     u_temp2, i_temp2 = self.user_rep.clone(), self.item_rep.clone
        #     u_temp.detach()
        #     i_temp.detach()
        #     u_temp2.detach()
        #     i_temp2.detach()
        #     u_temp2 = self.mlp(u_temp2)
        #     i_temp2 = self.mlp(i_temp2)
        #     u_temp = F.dropout(u_temp, self.dropout)
        #     i_temp = F.dropout(i_temp, self.dropout)
        # mask_loss_u = 1 - F.cosine_similarity(u_temp, u_temp2).mean()
        # mask_loss_i = 1 - F.cosine_similarity(i_temp, i_temp2).mean()
        # mask_f_loss = self.mask_weight_f * (mask_loss_i + mask_loss_u)

        # guide
        # 粗粒度-分布
        #r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean = self.fit_Gaussian_dis()
        # id and v+t
        # dis_loss_i_vt = (torch.abs(g_var - r_var) +
        #                  torch.abs(g_mean - r_mean)).mean()
        # # id and v
        # dis_loss_i_v = (torch.abs(g_var - v_var) +
        #                 torch.abs(g_mean - v_mean)).mean()
        # # id and t
        # dis_loss_i_t = (torch.abs(g_var - t_var) +
        #                 torch.abs(g_mean - t_mean)).mean()
        #
        # # v and v+t
        # dis_loss_v_vt = (torch.abs(r_var - v_var) +
        #                  torch.abs(r_mean - v_mean)).mean()
        #
        # # t and v+t
        # dis_loss_t_vt = (torch.abs(r_var - t_var) +
        #                  torch.abs(r_mean - t_mean)).mean()
        #
        # # v and t
        # dis_loss_v_t = (torch.abs(v_var - t_var) +
        #                 torch.abs(v_mean - t_mean)).mean()
        #
        # # total
        # dis_loss = (dis_loss_i_vt + dis_loss_i_v + dis_loss_i_t
        #             + dis_loss_v_vt + dis_loss_t_vt
        #             + dis_loss_v_t)

        # level4
        # dis_loss = (dis_loss_v_t)
        # level3
        # dis_loss = (dis_loss_v_vt + dis_loss_t_vt)
        # level2
        # dis_loss = (dis_loss_i_v + dis_loss_i_t)
        # level1
        # dis_loss = dis_loss_i_vt

        # align_loss = ((torch.abs(g_var - r_var) +
        #                  torch.abs(g_mean - r_mean)).mean() +
        #                 (torch.abs(g_var - v_var) +
        #                  torch.abs(g_mean - v_mean)).mean() +
        #                (torch.abs(g_var - t_var) +
        #                 torch.abs(g_mean - t_mean)).mean() +
        #               (torch.abs(r_var - v_var) +
        #                torch.abs(r_mean - v_mean)).mean() +
        #              (torch.abs(r_var - t_var) +
        #               torch.abs(r_mean - t_mean)).mean() +
        #             (torch.abs(v_var - t_var) +
        #              torch.abs(v_mean - t_mean)).mean())

        # 图噪音cl
        #inspired by SimGCL
        # mask_g_loss = (self.InfoNCE(self.result_embed_n1[:self.n_users], self.result_embed_n2[:self.n_users], self.temp)
        #                + self.InfoNCE(self.result_embed_n1[self.n_users:], self.result_embed_n2[self.n_users:], self.temp))

        # mask_g_loss = mask_g_loss * self.mask_weight_g

        # align_loss = align_loss * self.align_weight

        return loss_value + reg_loss #+ align_loss + mask_g_loss + mask_f_loss #+ um_loss

        # return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    def print_embd(self):
        return self.result_embed_v, self.result_embed_t


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h).cuda()
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        # h_2 = self.conv_embed_1(h_1, edge_index)

        x_hat = x + h + h_1
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

