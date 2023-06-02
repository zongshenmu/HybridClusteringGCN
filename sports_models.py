import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import SparseTensor, mul_nnz, matmul, mul, coalesce, matmul
import torch_geometric
import numpy as np

from utility.parser import parse_args

args = parse_args()

device=torch.device(f'cuda:{args.gpu_id}')

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def add_sparse(a, b):
    assert a.sizes() == b.sizes(), "The Tensor dimensions do not match"
    row_a, col_a, values_a = a.coo()
    row_b, col_b, values_b = b.coo()

    index = torch.stack([torch.cat([row_a, row_b]), torch.cat([col_a, col_b])])
    value = torch.cat([values_a, values_b])

    m, n = a.sizes()
    index, value = coalesce(index, value, m=m, n=n)
    res = SparseTensor.from_edge_index(index, value, sparse_sizes=(m, n))
    return res

def sub_sparse(a, b):
    assert a.sizes() == b.sizes(), "The Tensor dimensions do not match"
    row_a, col_a, values_a = a.coo()
    row_b, col_b, values_b = b.coo()

    index = torch.stack([torch.cat([row_a, row_b]), torch.cat([col_a, col_b])])
    value = torch.cat([values_a, -1 * values_b])

    m, n = a.sizes()
    index, value = coalesce(index, value, m=m, n=n)
    res = SparseTensor.from_edge_index(index, value, sparse_sizes=(m, n))
    return res

class LATTICE(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats,):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        # nn.init.normal_(self.user_embedding.weight, std=0.1)
        # nn.init.normal_(self.item_id_embedding.weight, std=0.1)

        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        if os.path.exists('../data/%s/%s-core/image_adj_%d.pt' % (args.dataset, args.core, args.topk)):
            image_adj = torch.load('../data/%s/%s-core/image_adj_%d.pt' % (args.dataset, args.core, args.topk))
        else:
            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_neighbourhood(image_adj, topk=args.topk)
            image_adj = compute_normalized_laplacian(image_adj)
            torch.save(image_adj, '../data/%s/%s-core/image_adj_%d.pt' % (args.dataset, args.core, args.topk))

        if os.path.exists('../data/%s/%s-core/text_adj_%d.pt' % (args.dataset, args.core, args.topk)):
            text_adj = torch.load('../data/%s/%s-core/text_adj_%d.pt' % (args.dataset, args.core, args.topk))
        else:
            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_neighbourhood(text_adj, topk=args.topk)
            text_adj = compute_normalized_laplacian(text_adj)
            torch.save(text_adj, '../data/%s/%s-core/text_adj_%d.pt' % (args.dataset, args.core, args.topk))

        self.text_original_adj = SparseTensor.from_dense(text_adj).to(device)
        self.image_original_adj = SparseTensor.from_dense(image_adj).to(device)

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        # weight = torch.Tensor([0.5, 0.5])
        # original_adj = add_sparse(mul_nnz(self.image_original_adj, weight[0]),
        #                           mul_nnz(self.text_original_adj, weight[1]))
        # self.item_adj = original_adj

        # self.num_clusters = 60
        # self.vcluster_pred = nn.Sequential(
        #     nn.Linear(args.feat_embed_dim, args.feat_embed_dim // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(args.feat_embed_dim // 2),
        #     nn.Linear(args.feat_embed_dim // 2, self.num_clusters),
        # )
        # self.tcluster_pred = nn.Sequential(
        #     nn.Linear(args.feat_embed_dim, args.feat_embed_dim // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(args.feat_embed_dim // 2),
        #     nn.Linear(args.feat_embed_dim // 2, self.num_clusters),
        # )
        # self.vc_emb = nn.Parameter(torch.Tensor(self.num_clusters, self.embedding_dim))
        # self.tc_emb = nn.Parameter(torch.Tensor(self.num_clusters, self.embedding_dim))
        # nn.init.xavier_uniform_(self.vc_emb)
        # nn.init.xavier_uniform_(self.tc_emb)
        self.cluster_softmax = nn.Softmax(dim=-1)

        self.logit_scale = nn.Parameter(torch.ones([1]))

        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        if args.cf_model == 'imp' or args.cf_model == 'imp_sc':
            self.group = 4
            self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
            self.fc2 = nn.Linear(self.embedding_dim, self.group)

        if args.cf_model == 'sc' or args.cf_model == 'imp_sc':
            dir_pth = f'../data/{args.dataset}/{args.core}-core'
            self.group = 4
            self.co_clusters = torch.FloatTensor(
                np.load(os.path.join(dir_pth, 'partitions', f'metis_{self.group}_ui2subg.npy'))).to(device)
            self.init = True
            self.dynamic = False
            if self.dynamic:
                self.co_fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
                self.co_fc2 = nn.Linear(self.embedding_dim, self.group)
                self.c2v_transform = nn.Linear(self.group, self.embedding_dim)
                self.adjust_weights = nn.Parameter(torch.Tensor([0.5, 0.5]))

    def forward(self, adj, build_item_graph=False):
        # image_feats = self.image_trs(self.image_embedding.weight)
        # text_feats = self.text_trs(self.text_embedding.weight)

        # vh = image_feats
        # vc = self.cluster_softmax(self.vcluster_pred(vh))
        # self.vc_feats = torch.matmul(vc, self.vc_emb)
        # th = text_feats
        # tc = self.cluster_softmax(self.tcluster_pred(th))
        # self.tc_feats = torch.matmul(tc, self.tc_emb)

        image_item_embeds = self.item_id_embedding.weight
        text_item_embeds = self.item_id_embedding.weight
        for i in range(args.n_layers):
            image_item_embeds = matmul(self.image_original_adj, image_item_embeds)
        self.image_item_embeds=image_item_embeds
        for i in range(args.n_layers):
            text_item_embeds = matmul(self.text_original_adj, text_item_embeds)
        self.text_item_embeds=text_item_embeds

        # att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
        # weight = self.cluster_softmax(att)
        # h = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(dim=1) * text_item_embeds

        h=0.5*image_item_embeds+0.5*text_item_embeds
        self.h=h
        # h += 0.5 * self.vc_feats + 0.5 * self.tc_feats

        if args.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)
        elif args.cf_model == 'imp':
            ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)

            adj=SparseTensor.from_torch_sparse_coo_tensor(adj)
            user_group_embeddings_side = matmul(adj, ego_embeddings) + ego_embeddings

            user_group_embeddings_side = F.leaky_relu(self.fc1(user_group_embeddings_side))
            user_group_embeddings_side = F.dropout(user_group_embeddings_side, 1 - 0.6)
            user_group_embeddings_sum = self.fc2(user_group_embeddings_side)

            a_top, a_top_idx = torch.topk(user_group_embeddings_sum, 1)
            user_group_embeddings = torch.eq(user_group_embeddings_sum, a_top).type_as(user_group_embeddings_sum)
            u_group_embeddings, i_group_embeddings = torch.split(user_group_embeddings, [self.n_users, self.n_items])
            i_group_embeddings = torch.ones(i_group_embeddings.shape).type_as(i_group_embeddings)
            user_group_embeddings = torch.cat([u_group_embeddings, i_group_embeddings], dim=0)
            A_fold_hat_group = self._split_A_hat_group(adj, user_group_embeddings)

            all_embeddings = [ego_embeddings]
            side_embeddings = matmul(adj, ego_embeddings)
            all_embeddings += [side_embeddings]

            ego_embeddings_g = []
            for g in range(0, self.group):
                ego_embeddings_g.append(ego_embeddings)

            ego_embeddings_f = []
            for k in range(1, self.n_ui_layers):
                for g in range(0, self.group):
                    side_embeddings = matmul(A_fold_hat_group[g], ego_embeddings_g[g])
                    ego_embeddings_g[g] = ego_embeddings_g[g] + side_embeddings
                    if k == 1:
                        ego_embeddings_f.append(matmul(adj, side_embeddings))
                    else:
                        ego_embeddings_f[g] = matmul(adj, side_embeddings)
                ego_embeddings = torch.sum(torch.stack(ego_embeddings_f, dim=0), dim=0, keepdim=False)
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = torch.mean(all_embeddings, dim=1)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'sc':
            self.adj = SparseTensor.from_torch_sparse_coo_tensor(adj)
            if self.init:
                for k in range(0, self.group):
                    tmp = self.co_clusters[:, k]
                    tmp_intra = mul(mul(self.adj, tmp.unsqueeze(0)), tmp.unsqueeze(1))
                    if k == 0:
                        self.intra_adj = tmp_intra
                    else:
                        self.intra_adj = add_sparse(self.intra_adj, tmp_intra)
                self.inter_adj = sub_sparse(self.adj, self.intra_adj)
                self.init = False

            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            # cluster gcn
            list_local_embeddings = [ego_embeddings]
            for k in range(0, self.n_ui_layers):
                side_embeddings = matmul(self.intra_adj, ego_embeddings)
                ego_embeddings = side_embeddings
                list_local_embeddings += [ego_embeddings]
            local_embeddings = torch.mean(torch.stack(list_local_embeddings, dim=1), dim=1)
            global_embeddings = matmul(self.inter_adj, local_embeddings)
            all_embeddings = global_embeddings + local_embeddings
            if self.dynamic:
                # dynamic cluster
                group_embeddings = F.leaky_relu(self.co_fc1(all_embeddings))
                group_embeddings = F.dropout(group_embeddings, 1 - 0.6)
                group_embeddings = torch.sigmoid(self.co_fc2(group_embeddings))
                weights = self.cluster_softmax(self.adjust_weights)
                group_scores = weights[0] * self.co_clusters + weights[1] * group_embeddings
                self.ui_clusters, a_top_idx = torch.topk(group_scores, 1)
                self.co_clusters = torch.eq(group_scores, self.ui_clusters).type_as(group_scores)
                self.init = True

            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings


    def _split_A_hat_group(self, X, group_embedding):
        group_embedding = group_embedding.T
        A_fold_hat_group = []
        # k groups
        for k in range(0, self.group):
            A_fold_hat_item=mul(mul(X,group_embedding[k].unsqueeze_(0)),group_embedding[k].unsqueeze_(1))
            A_fold_hat_group.append(A_fold_hat_item)
        return A_fold_hat_group

    def batched_contrastive_loss(self, z1, z2, batch_size=4096, tau=0.5):
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def vt_contrastive_loss(self,vfeats,tfeats):

        logits=self.sim(vfeats,tfeats)*self.logit_scale.exp()
        # tau=0.07
        # logits = (self.sim(vfeats, tfeats) / tau).exp()
        labels=torch.arange(vfeats.shape[0]).to(device)
        lossv=F.cross_entropy(logits, labels)
        losst=F.cross_entropy(logits.t(), labels)
        return (lossv+losst)/2
