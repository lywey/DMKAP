import pickle
import os
from functools import reduce


# number of unique ICD codes
ICD_NUM = 4880


def process_newTrees(dataseqs, tree_old):
    leaf2tree = pickle.load(open('./build_trees/mimic.level5.pk', 'rb'))
    trees_l4 = pickle.load(open('./build_trees/mimic.level4.pk', 'rb'))
    trees_l3 = pickle.load(open('./build_trees/mimic.level3.pk', 'rb'))
    trees_l2 = pickle.load(open('./build_trees/mimic.level2.pk', 'rb'))
    tree_seq = []

    leaf2tree.update(trees_l4)
    leaf2tree.update(trees_l3)
    leaf2tree.update(trees_l2)

    for patient in dataseqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in leaf2tree[code]:
                    leaf2tree[code].remove(code)
                    newVisit.append(leaf2tree[code])
                else:
                    newVisit.append(leaf2tree[code])
            newVisit = list(set(reduce(lambda x,y:x+y, newVisit)))
            newPatient.append(newVisit)
        tree_seq.append(newPatient)

    newTreeseq = []
    for patient in tree_seq:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                newVisit.append(tree_old[code])
            newPatient.append(newVisit)
        newTreeseq.append(newPatient)

    return newTreeseq


if __name__ == '__main__':
    fileName = 'mimic_trees'
    filePath = './data/'
    outFile = filePath + fileName

    data_seqs = pickle.load(open('./build_trees/mimic.seqs', 'rb'))
    trees_type = pickle.load(open('./build_trees/mimic.types', 'rb'))
    retype = dict([(v, k) for k, v in trees_type.items()])

    treenode = {}
    tree2old = {}
    count = 0

    # Ancestor node index from 4880 to 5608 (728 in total)
    for i in range(ICD_NUM, len(retype)):
        treenode[count] = retype[i]
        tree2old[i] = count
        count += 1

    newTreeSeq = process_newTrees(data_seqs, tree2old)

    if not os.path.exists(filePath):
        os.mkdir(filePath)
    pickle.dump(newTreeSeq, open(outFile + '.seqs', 'wb'), -1)
