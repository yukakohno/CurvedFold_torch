# -*- coding: utf-8 -*-
"""
@author: ykohno
"""

import csv

def ReadControlPoints( path ):
    """
    example:
    10 1 # file type, mode: B, kv,tr,fa,k2d
    7 # plot count
    0.000000	0.005000	1.570796	0.000000
    0.000000	0.004500	1.291544	0.004536
    0.000000	0.004000	1.117011	0.005988
    0.000000	0.003500	1.064651	0.007399
    0.000000	0.003000	1.121374	0.005987
    0.000000	0.002500	1.291544	0.004538
    0.000000	0.002000	1.570796	0.000000
    """
    
    with open(path) as f:
        reader = csv.reader(f,delimiter='\t')
        l = [row for row in reader]
    
    P_kv = []
    P_tr = []
    P_fa = []
    P_k2d = []
    for row in l[2:]:
        P_kv.append(float(row[0]))
        P_tr.append(float(row[1]))
        P_fa.append(float(row[2]))
        P_k2d.append(float(row[3]))
        
    return P_kv, P_tr, P_fa, P_k2d

def Readm2m3( path ):
    """
    example:
    m2
    0.884988	-0.465615	0.000000
    0.465615	0.884988	0.000000
    10.156250	89.062500	1.000000
    m3
    0.884988	-0.465615	0.000000	0.000000
    0.465615	0.884988	0.000000	0.000000
    0.000000	0.000000	1.000000	0.000000
    10.156250	89.062500	0.000000	1.000000
    """
    
    with open(path) as f:
        reader = csv.reader(f,delimiter='\t')
        l = [row for row in reader]
    
    m2 = l[1:4]
    m3 = l[5:9]
    m2 = [[float(x) for x in y] for y in m2]
    m3 = [[float(x) for x in y] for y in m3]

    return m2, m3

def ReadTarget( path ):
    """
    example:
    22.643123 24.016077 21.951300 20.000000 20.000000
    42.346042 23.623047 18.539481 40.000000 20.000000
    62.048961 23.230018 15.127662 60.000000 20.000000
    81.776372 22.957820 12.191825 80.000000 20.000000
    """
    with open(path) as f:
        reader = csv.reader(f,delimiter=' ')
        l = [row for row in reader]
    
    tgt = []
    org_cp = []
    for row in l:
        tgt.append([float(row[0]),float(row[1]),float(row[2])])
        org_cp.append([float(row[3]),float(row[4])])
    
    return tgt, org_cp

def ReadTargetParam( path ):
    
    with open(path) as f:
        reader = csv.reader(f,delimiter=',')
        l = [row for row in reader]
    
    tgtK = []
    tgtT = []
    tgtA = []
    for row in l[1:]:
        tgtK.append([float(row[12])])
        tgtT.append([float(row[13])])
        tgtA.append([float(row[21])])
    
    return tgtK, tgtT, tgtA

def ReadTargetX( path ):
    
    with open(path) as f:
        reader = csv.reader(f,delimiter=',')
        l = [row for row in reader]
    
    tgtX = []
    for row in l[1:]:
        tgtX.append([float(row[0]),float(row[1]),float(row[2])])
    
    return tgtX

def ReadTargetXTNB( path ):
    
    with open(path) as f:
        reader = csv.reader(f,delimiter=',')
        l = [row for row in reader]
    
    tgtX = []
    tgtT = []
    tgtN = []
    tgtB = []
    for row in l[1:]:
        tgtX.append([float(row[0]),float(row[1]),float(row[2])])
        tgtT.append([float(row[3]),float(row[4]),float(row[5])])
        tgtN.append([float(row[6]),float(row[7]),float(row[8])])
        tgtB.append([float(row[9]),float(row[10]),float(row[11])])
    
    return tgtX, tgtT, tgtN, tgtB

def ReadTargetRuling( path ):
    
    with open(path) as f:
        reader = csv.reader(f,delimiter=',')
        l = [row for row in reader]
    
    tgtBetaR = []
    tgtBetaL = []
    for row in l[1:]:
        tgtBetaR.append([float(row[26])])
        tgtBetaL.append([float(row[27])])

    return tgtBetaR, tgtBetaL
