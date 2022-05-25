import pickle
import dgl
import os
import networkx as nx
import scipy.sparse as sp


# number of unique ICD codes + number of knowledge ancestor node
# 4880 + 728
ALL_CODES_NUM = 5608


def tree_levelall():
    leaf2tree = pickle.load(open('./build_trees/mimic.level5.pk', 'rb'))
    trees_l4 = pickle.load(open('./build_trees/mimic.level4.pk', 'rb'))
    trees_l3 = pickle.load(open('./build_trees/mimic.level3.pk', 'rb'))
    trees_l2 = pickle.load(open('./build_trees/mimic.level2.pk', 'rb'))

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    return leaf2tree


if __name__ == '__main__':
    fileName = 'mimic_gcn'
    filePath = './data/'
    outFile = filePath + fileName

    types = pickle.load(open('./build_trees/mimic.types', 'rb'))
    retype = dict([(v, k) for k, v in types.items()])

    edgelist = {}
    tree = tree_levelall()

    for key, value in tree.items():
        for index, node in enumerate(value):
            if index == 0:
                continue
            if index == len(value) - 1:
                if node not in edgelist:
                    edgelist[node] = [key]
                else:
                    edgelist[node].append(key)
            else:
                if node not in edgelist:
                    edgelist[node] = [value[index+1]]
                elif value[index+1] not in edgelist[node]:
                    edgelist[node].append(value[index+1])

    # Source nodes for edges
    src_ids = []
    # Destination nodes for edges
    dst_ids = []

    for key, value in edgelist.items():
        src_ids.extend([key for i in range(len(value))])
        dst_ids.extend(value)

    temp = src_ids + dst_ids
    code = list(set(temp))
    all = [ i for i in range(ALL_CODES_NUM)]
    miss = list(set(all).difference(set(code)))
    misscode = []
    for i in miss:
        misscode.append(retype[i])

    srcc = []
    dstt = []
    for code in misscode:
        tokens = code.strip().split('.')
        prefix = ''
        for i in range(len(tokens)-1):
            if i == 0 or i == len(tokens)-1:
                prefix = prefix + tokens[i]
            else:
                prefix = prefix + '.' + tokens[i]
        srcc.append(int(types[prefix]))
        dstt.append(int(types[code]))

    src_ids.extend(srcc)
    dst_ids.extend(dstt)

    g = dgl.graph((src_ids, dst_ids))
    # Directed graph to undirected graph
    bg = dgl.to_bidirected(g)

    nx_G = bg.to_networkx().to_undirected()
    N = len(nx_G)
    adj = nx.to_numpy_array(nx_G)
    adj = sp.coo_matrix(adj)

    if not os.path.exists(filePath):
        os.mkdir(filePath)
    pickle.dump(adj, open(outFile + '.adj', 'wb'), -1)

