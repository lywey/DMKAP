import pickle
import os


# number of unique ICD codes
ICD_NUM = 4880


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr


if __name__ == '__main__':
    dxref = './$dxref 2015.csv'
    seqs = pickle.load(open('./build_trees/mimic.seqs', 'rb'))
    types = pickle.load(open('./build_trees/mimic.types', 'rb'))
    fileName = 'mimic'
    filePath = './data/'
    outFile = filePath + fileName

    retype = dict(sorted([(v, k) for k, v in types.items()]))

    # icd codes are grouped by CCS single-level diagnosis grouper
    ref = {}
    infd = open(dxref, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().replace('\'', '').split(',')
        icd9 = 'D_' + convert_to_icd9(tokens[0].replace(' ', ''))
        ccs = int(tokens[1].replace(' ', ''))
        ref[icd9] = ccs
    infd.close()

    temp = {}  # key:icd code index(0-4879), value:ccs classification code
    category = []
    for k, v in retype.items():
        if k == ICD_NUM: break
        temp[k] = ref[v]
        if ref[v] not in category:
            category.append(ref[v])

    length = len(category)

    ccsMap = {}  # key:ccs classification code, value: index(0-271)
    for i in range(len(category)):
        ccsMap[category[i]] = i

    indexMap = {}  # key:icd code index(0-4879), value: ccs index(0-271)
    for k, v in temp.items():
        indexMap[k] = ccsMap[v]

    newLabel = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if indexMap[code] not in newVisit:
                    newVisit.append(indexMap[code])
            newPatient.append(newVisit)
        newLabel.append(newPatient)

    if not os.path.exists(filePath):
        os.mkdir(filePath)
    pickle.dump(newLabel, open(outFile + '.labels', 'wb'), -1)
