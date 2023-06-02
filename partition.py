import os

import dgl
import scipy.sparse as spsp
import numpy as np

def dgl_metis(dir_pth,pnum):
    adj = spsp.load_npz(dir_pth + '/s_norm_adj_mat.npz')
    g = dgl.graph(adj.nonzero())
    dataset=dir_pth.split('/')[-2]
    outdir=os.path.join(dir_pth,'partitions','dglmetis')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(f'{outdir}/{pnum}parts/{dataset}.json'):
        dgl.distributed.partition_graph(g, dataset, pnum, num_hops=1, part_method='metis',
                                        out_path=f'{outdir}/{pnum}parts', reshuffle=False,
                                        balance_edges=True)
    npid=[]
    for i in range(pnum):
        g, vf, ef, gpb, _, _, _ = \
            dgl.distributed.load_partition( f'{outdir}/{pnum}parts/{dataset}.json', i)
        # if i == 0:
        #     print(gpb.metadata())
        # print(g)
        for k,v in vf.items():
            print(k,v)
        for k,v in ef.items():
            print(k,v)

        ori_nodes = gpb.partid2nids(gpb.partid).numpy()
        tmp=np.vstack([ori_nodes,np.ones(ori_nodes.shape)*i])
        # print(tmp.shape,tmp)
        npid.append(tmp)
    nids=np.hstack(npid)
    # print(nids.shape)
    sort_indices=np.argsort(nids[0])
    v2p_sort=nids[1][sort_indices]
    res = np.eye(pnum)[v2p_sort.astype(int)]
    np.save(os.path.join(dir_pth,'partitions',f'metis_{pnum}_ui2subg.npy'),res)


if __name__=='__main__':
    core = '5-core'
    datasets = ['clothing']
    # datasets=['clothing','sports','baby']
    for dataset in datasets:
        dir = f'../data/{dataset}/{core}'
        ps = [2, 6, 8, 10]
        for i in ps:
            dgl_metis(dir, i)
