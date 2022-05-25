#-*- coding: UTF-8 -*-
# Execute the below command
# python build_trees.py ccs_multi_dx_tool_2015.csv ./process_mimic/mimic.seqs ./process_mimic/mimic.types
import sys
import pickle
import os


if __name__ == '__main__':
    infile = sys.argv[1]  # ccs_multi_dx_tool_2015.csv
    seqFile = sys.argv[2]
    typeFile = sys.argv[3]
    fileName = 'mimic'
    filePath = './build_trees/'
    outFile = filePath + fileName

    infd = open(infile, 'r')
    _ = infd.readline()

    seqs = pickle.load(open(seqFile, 'rb'))
    types = pickle.load(open(typeFile, 'rb'))

    startSet = set(types.keys())
    hitList = []
    missList = []
    cat1count = 0
    cat2count = 0
    cat3count = 0
    cat4count = 0
    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()
        
        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types: 
            missList.append(icd9)
        else: 
            hitList.append(icd9)

        if cat1 not in types: 
            cat1count += 1
            types[cat1] = len(types)

        if len(cat2) > 0:
            if cat2 not in types: 
                cat2count += 1
                types[cat2] = len(types)
        if len(cat3) > 0:
            if cat3 not in types: 
                cat3count += 1
                types[cat3] = len(types)
        if len(cat4) > 0:
            if cat4 not in types: 
                cat4count += 1
                types[cat4] = len(types)
    infd.close()

    rootCode = len(types)
    types['A_ROOT'] = rootCode
    print(rootCode)

    print('cat1count: %d' % cat1count)
    print('cat2count: %d' % cat2count)
    print('cat3count: %d' % cat3count)
    print('cat4count: %d' % cat4count)
    print('Number of total ancestors: %d' % (cat1count + cat2count + cat3count + cat4count + 1))
    print('hit count: %d' % len(set(hitList)))
    print('miss count: %d' % len(startSet - set(hitList)))
    missSet = startSet - set(hitList)

    if not os.path.exists(filePath):
        os.mkdir(filePath)

    pickle.dump(types, open(outFile + '.types', 'wb'), -1)  # types contains icd9 codes and ancestor nodes

    fiveMap = {}
    fourMap = {}
    threeMap = {}
    twoMap = {}
    oneMap = dict([(types[icd], [types[icd], rootCode]) for icd in missSet])

    infd = open(infile, 'r')
    infd.readline()

    for line in infd:
        tokens = line.strip().split(',')
        icd9 = tokens[0][1:-1].strip()
        cat1 = tokens[1][1:-1].strip()
        desc1 = 'A_' + tokens[2][1:-1].strip()
        cat2 = tokens[3][1:-1].strip()
        desc2 = 'A_' + tokens[4][1:-1].strip()
        cat3 = tokens[5][1:-1].strip()
        desc3 = 'A_' + tokens[6][1:-1].strip()
        cat4 = tokens[7][1:-1].strip()
        desc4 = 'A_' + tokens[8][1:-1].strip()

        if icd9.startswith('E'):
            if len(icd9) > 4: icd9 = icd9[:4] + '.' + icd9[4:]
        else:
            if len(icd9) > 3: icd9 = icd9[:3] + '.' + icd9[3:]
        icd9 = 'D_' + icd9

        if icd9 not in types: continue
        icdCode = types[icd9]

        codeVec = []

        if len(cat4) > 0:
            code4 = types[cat4]
            code3 = types[cat3]
            code2 = types[cat2]
            code1 = types[cat1]
            fiveMap[icdCode] = [icdCode, rootCode, code1, code2, code3, code4]
        elif len(cat3) > 0:
            code3 = types[cat3]
            code2 = types[cat2]
            code1 = types[cat1]
            fourMap[icdCode] = [icdCode, rootCode, code1, code2, code3]
        elif len(cat2) > 0:
            code2 = types[cat2]
            code1 = types[cat1]
            threeMap[icdCode] = [icdCode, rootCode, code1, code2]
        else:
            code1 = types[cat1]
            twoMap[icdCode] = [icdCode, rootCode, code1]

    pickle.dump(fiveMap, open(outFile + '.level5.pk', 'wb'), -1)
    pickle.dump(fourMap, open(outFile + '.level4.pk', 'wb'), -1)
    pickle.dump(threeMap, open(outFile + '.level3.pk', 'wb'), -1)
    pickle.dump(twoMap, open(outFile + '.level2.pk', 'wb'), -1)
    pickle.dump(oneMap, open(outFile + '.level1.pk', 'wb'), -1)
    pickle.dump(seqs, open(outFile + '.seqs', 'wb'), -1)