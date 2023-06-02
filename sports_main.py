import os
from utility.parser import parse_args

args = parse_args()

from time import time
import math
import random
import sys
import torch.nn.functional as F
import torch.optim as optim

from sports_models import LATTICE

from utility.batch_test import *


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu


set_seed(args.seed)


class Trainer(object):
    def __init__(self, data_config):
        # argument settings
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.model_name = args.model_name
        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.norm_adj = data_config['norm_adj']
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(device)

        image_feats = np.load('../data/{}/image_feat.npy'.format(args.dataset))
        text_feats = np.load('../data/{}/text_feat.npy'.format(args.dataset))

        self.model = LATTICE(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, image_feats,
                             text_feats)
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = self.set_lr_scheduler()

        self.save_name = f"outputs/{args.dataset}_{args.cf_model.lower()}"
        if not os.path.exists(self.save_name):
            os.makedirs(self.save_name)

        if args.eval:
            model_dict = torch.load(os.path.join(self.save_name, 'best_model.pth'))
            self.model.load_state_dict(model_dict)

    def set_lr_scheduler(self):
        fac = lambda epoch: 0.96 ** (epoch / 50)
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        return scheduler

    def test(self, users_to_test, is_val):
        self.model.eval()
        with torch.no_grad():
            ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=True)
        result = test_torch(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):
        training_time_list = []
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        cur_best_pre_0 = 0.

        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        if args.set_loss:
            data_generator.init_pos_and_neg_sets()
        for epoch in (range(args.epoch)):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            f_time, b_time, loss_time, opt_time, clip_time, emb_time = 0., 0., 0., 0., 0., 0.
            sample_time = 0.
            build_item_graph = True
            if args.set_loss:
                neg_sum = 3
                pos_num_sel = 3  # np.random.randint(2, neg_sum)  # np.random.randint(2,neg_sum+1) 3
                iters = data_generator.get_iter_batch_data(pos_num_sel, neg_sum)
            else:
                iters = range(n_batch)
            for iter_data in iters:
                self.model.train()
                self.optimizer.zero_grad()
                sample_t1 = time()
                if args.set_loss:
                    users, pos_items, neg_items = iter_data
                    users = users.to(device)
                    pos_items = pos_items.to(device)
                    neg_items = neg_items.to(device)
                else:
                    idx = iter_data
                    users, pos_items, neg_items = data_generator.sample()
                sample_time += time() - sample_t1
                ua_embeddings, ia_embeddings = self.model(self.norm_adj, build_item_graph=build_item_graph)
                build_item_graph = False
                u_g_embeddings = ua_embeddings[users]
                pos_i_g_embeddings = ia_embeddings[pos_items]
                neg_i_g_embeddings = ia_embeddings[neg_items]

                # batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                #                                                               neg_i_g_embeddings)
                if args.set_loss:
                    batch_mf_loss, batch_emb_loss, batch_reg_loss = self.multi_setrank_loss(u_g_embeddings,
                                                                                            pos_i_g_embeddings,
                                                                                            neg_i_g_embeddings)
                    # batch_reg_loss = 0.03 * self.model.vt_contrastive_loss(self.model.image_item_embeds,
                    #                                                        self.model.text_item_embeds)
                    image_embs = torch.cat(
                        [self.model.image_item_embeds[pos_items], self.model.image_item_embeds[neg_items]], dim=1).view(
                        pos_items.shape[0], -1)
                    text_embs = torch.cat(
                        [self.model.text_item_embeds[pos_items], self.model.text_item_embeds[neg_items]], dim=1).view(
                        pos_items.shape[0], -1)
                    batch_reg_loss = self.model.vt_contrastive_loss(image_embs, text_embs)
                    batch_reg_loss = 0.03 * batch_reg_loss
                else:
                    batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)
                    # img_embs = torch.cat(
                    #     [self.model.image_item_embeds[pos_items],
                    #      self.model.image_item_embeds[neg_items]], dim=0).view(len(pos_items), -1)
                    # txt_embs = torch.cat(
                    #     [self.model.text_item_embeds[pos_items],
                    #      self.model.text_item_embeds[neg_items]], dim=0).view(len(neg_items), -1)
                    # # img_clu = torch.cat(
                    # #     [self.model.vc_feats[pos_items],
                    # #      self.model.vc_feats[neg_items]], dim=0).view(len(pos_items), -1)
                    # # txt_clu = torch.cat(
                    # #     [self.model.tc_feats[pos_items],
                    # #      self.model.tc_feats[neg_items]], dim=0).view(len(neg_items), -1)
                    # # img_embs,txt_embs,\
                    # # img_clu,txt_clu=self.model.image_item_embeds,self.model.text_item_embeds,\
                    # #                 self.model.vc_feats,self.model.tc_feats
                    # # batch_reg_loss=self.model.vt_contrastive_loss(img_embs,txt_embs)
                    # tmp = torch.cat(
                    #     [self.model.h[pos_items],
                    #      self.model.h[neg_items]], dim=0).view(len(neg_items), -1)
                    # batch_reg_loss = self.model.vt_contrastive_loss(img_embs, tmp)
                    # batch_reg_loss += self.model.vt_contrastive_loss(txt_embs, tmp)
                    # # batch_reg_loss +=self.model.vt_contrastive_loss(img_clu, txt_clu)
                    # # batch_reg_loss += self.model.batched_contrastive_loss(img_embs,txt_embs)
                    # # batch_reg_loss += self.model.batched_contrastive_loss(txt_embs, img_embs)
                    # batch_reg_loss = 0.03 * batch_reg_loss
                    # lambda_ = 1e-3
                    # neighbor_embeds = ia_embeddings[res_mat[pos_items]]  # len(pos_items) * num_neighbors * dim
                    # sim_scores = res_sim_mat[pos_items].to(device)
                    # user_embeds = ua_embeddings[users].unsqueeze(1)
                    # ii_loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
                    # batch_reg_loss += lambda_ * ii_loss.sum()

                batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss

                batch_loss.backward()
                self.optimizer.step()

                loss += float(batch_loss)
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)

            self.lr_scheduler.step()

            del ua_embeddings, ia_embeddings, u_g_embeddings, neg_i_g_embeddings, pos_i_g_embeddings

            if math.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            print(f'Epoch {epoch} train loss: {loss:.4f}')

            if epoch % args.verbose != 0:
                continue

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            # ret = self.test(users_to_val, is_val=True)
            ret = self.test(users_to_test, is_val=False)
            training_time_list.append(t2 - t1)

            if args.verbose > 0:
                print(
                    f"\t val recall: {ret['recall'][-1]:.4f}, precision: {ret['precision'][-1]:.4f}, ndcg: {ret['ndcg'][-1]:.4f}")

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = ret
                stopping_step = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_name, 'best_model.pth'))
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                print('#####Early stopping steps: %d #####' % stopping_step)
            else:
                print('#####Early stop! #####')
                break

        # print(test_ret)
        print(
            f"All epochs test recall: {test_ret['recall'][-1]:.4f}, precision: {test_ret['precision'][-1]:.4f}, ndcg: {test_ret['ndcg'][-1]:.4f}")

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def multi_setrank_loss(self, user_embs, pos_item_embs, neg_item_embs):
        set_pos_num = pos_item_embs.shape[1]
        all_pos_scores = []
        # multi pos items
        for i in range(set_pos_num):
            all_pos_scores.append(torch.sum(torch.mul(user_embs, pos_item_embs[:, i, :]), dim=1))
        neg_k0 = neg_item_embs[:, 0, :]
        uk0_score = torch.sum(torch.mul(user_embs, neg_k0), dim=1)
        all_pos_sim = []
        all_neg_sim = [self.fun_z(all_pos_scores[0], uk0_score, False)]
        for i in range(1, set_pos_num):
            # all_pos_sim.append(self.fun_z(all_pos_scores[0],all_pos_scores[i],True))
            all_pos_sim.append(self.fun_z(all_pos_scores[0], all_pos_scores[i], False))
            all_neg_sim.append(self.fun_z(all_pos_scores[i], uk0_score, False))
        rank_loss = sum(all_neg_sim)
        neg_min = sum(all_neg_sim)
        for k in range(1, neg_item_embs.shape[1]):
            neg_k = neg_item_embs[:, k, :]
            uk_score = torch.sum(torch.mul(user_embs, neg_k), dim=1)
            k_all_score = 0
            for i in range(set_pos_num):
                k_all_score += self.fun_z(all_pos_scores[i], uk_score, False)
            rank_loss += k_all_score
            neg_min = torch.where(k_all_score > neg_min, k_all_score, neg_min)
        pos_dis = sum(all_pos_scores) / (set_pos_num - 1)
        bin_labels = pos_dis.new_ones(pos_dis.size(), dtype=torch.int64)
        hard_loss = F.margin_ranking_loss(pos_dis, neg_min / set_pos_num, bin_labels, 1)
        mf_loss = hard_loss + 0.25 * rank_loss.mean()
        # mf_loss=self.model.set_contrastive_loss(user_embs, pos_item_embs, neg_item_embs)

        emb_loss = self.decay * (user_embs ** 2 + (pos_item_embs ** 2).sum(dim=1) + (
                neg_item_embs ** 2).sum(dim=1)).sum(dim=1).mean()

        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def fun_z(self, z_ai, z_aj, flag):
        if flag:  # pos pair
            distance_pair = torch.abs(z_ai - z_aj)
            result = torch.where(distance_pair > 0.5, torch.zeros_like(distance_pair) + 0.5, distance_pair)
            return result  # -(1-result).log()#torch.exp(result)
        else:
            distance_pair = F.softplus(-(z_ai - z_aj))  # -((z_ai - z_aj).sigmoid().log())
            return distance_pair

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    config['norm_adj'] = norm_adj
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    res_mat, res_sim_mat = data_generator.get_ii_constraint_mat(norm_adj)

    trainer = Trainer(data_config=config)
    if args.eval:
        users_to_test = list(data_generator.test_set.keys())
        trainer.test(users_to_test, is_val=False)
    else:
        trainer.train()
