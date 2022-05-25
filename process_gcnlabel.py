import pickle
import os
import numpy as np


# number of unique ICD codes
ICD_NUM = 4880


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr

def convert_num(dxStr):
    if len(dxStr) == 1: return int(dxStr[:])
    else :
        if dxStr[:2]=='A_': return int(0)
        return int(dxStr[:2].replace('.', ''))


if __name__ == '__main__':
    multidx = './ccs_multi_dx_tool_2015.csv'
    seqs = pickle.load(open('./build_trees/mimic.seqs', 'rb'))
    types = pickle.load(open('./build_trees/mimic.types', 'rb'))
    fileName = 'mimic_gcnlabel'
    filePath = './data/'
    outFile = filePath + fileName

    retype = dict(sorted([(v, k) for k, v in types.items()]))

    # icd codes are grouped by CCS multi-level diagnosis grouper
    ref = {}
    infd = open(multidx, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().replace('\'', '').split(',')
        icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
        multiccs = int(tokens[1].replace(' ', ''))
        ref[icd9] = multiccs
    infd.close()

    # icd codes index is 0-4879
    category = {}
    for i in range(ICD_NUM):
        category[i] = ref[retype[i]]

    for i in range(ICD_NUM, len(types)):
        category[i] = convert_num(retype[i])

    c1 = []
    for k in category.items():
        c1.append(k)
    labels = np.array(c1)

    classes = set(labels[:, -1])
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels[:, -1])), dtype=np.int32)

    if not os.path.exists(filePath):
        os.mkdir(filePath)
    np.save(outFile, labels_onehot)
