# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import os
# write the path of this directory 
#os.chdir('*/CurvedFold_py')
import numpy as np
import csv

import ReadFile 
import Spline
import ReconstructCurve
import TNBkt
import Ruling
import Polygon

"""
    Read File
"""
P_kv, P_tr, P_fa, P_k2d = ReadFile.ReadControlPoints('input/P_1.txt')
m2_, m3_ = ReadFile.Readm2m3('input/m2m3.txt')

P_kv = np.array(P_kv)
P_tr = np.array(P_tr)
P_fa = np.array(P_fa)
P_k2d = np.array(P_k2d)
m2 = np.array(m2_)
m3 = np.array(m3_)

"""
    Target parameters
"""
tgt, org_cp = ReadFile.ReadTarget('input/target.txt')

tgtK_, tgtT_, tgtA_ = ReadFile.ReadTargetParam( 'input/target.csv' )
tgtK = np.array(tgtK_)
tgtTR = np.array(tgtT_)
tgtA = np.array(tgtA_)
tgtX_, tgtT_, tgtN_, tgtB_ = ReadFile.ReadTargetXTNB( 'input/target.csv' )
tgtX = np.array(tgtX_)
tgtT = np.array(tgtT_)
tgtN = np.array(tgtN_)
tgtB = np.array(tgtB_)
tgtBetaR_, tgtBetaL_ = ReadFile.ReadTargetRuling( 'input/target.csv' )
tgtBetaR = np.array(tgtBetaR_)
tgtBetaL = np.array(tgtBetaL_)

"""
    Basic parameters
"""
Pcnt = len(P_kv)
Xcnt = 40
dX = 5.0
#Px = np.linspace(0, Pcnt-1, Pcnt)
Xx = np.linspace(0, Pcnt-1, Xcnt)
spmat, dt = Spline.get_coeff(Xx, Pcnt-1)

"""
    2D curve
"""
k2d = Spline.interpolate_spline_7(P_k2d, spmat, dt)
X2d, T2d, N2d = ReconstructCurve.ReconstructX2d(k2d, m2, dX)
T2d, N2d, k2d = TNBkt.calcTNk2d(X2d, dX)

"""
    3D curve
"""
#kv = Spline.interpolate_spline_7(P_kv, mat, dt)
tr = Spline.interpolate_spline_7(P_tr, spmat, dt)
fa = Spline.interpolate_spline_7(P_fa, spmat, dt)

cosa = np.cos(fa)
sina = np.sin(fa)
tana = np.tan(fa)
kv = k2d/cosa

X, T, N, B = ReconstructCurve.ReconstructX(kv, tr, m3, dX)
#X, T, N, B = ReconstructCurve.ReconstructX_grad(kv, tr, m3, dX)
T, N, B, kv, tr = TNBkt.calcTNBkt(X, dX)
#T, N, B, kv, tr = TNBkt.calcTNBkt_grad(X, dX)

"""
    rulings
"""
da = TNBkt.calcDiffAngle(fa, dX)
#da = TNBkt.calcDiffAngle_grad(fa, dX)
betaR, betaL, cosbR, cosbL, sinbR, sinbL = Ruling.RulingAngle(kv, tr, da, sina)
RulR, RulL, RulR2d, RulL2d = Ruling.Ruling(T, N, B, T2d, cosa, sina, cosbR, cosbL, sinbR, sinbL)
    
"""
    export all parameters
"""
header = ["X.x", "X.y", "X.z", "T.x", "T.y", "T.z",
          "N.x", "N.y", "N.z", "B.x", "B.y", "B.z", "kv", "tr",
          "X2d.x", "X2d.y", "T2d.x", "T2d.y", "N2d.x", "N2d.y", "k2d",
          "alpha", "da", "cosa", "sina", "tana",
          "betaR", "betaL", "cosbR", "cosbL", "sinbR", "sinbL",
          "RulR.x", "RulR.y", "RulR.z", "RulL.x", "RulL.y", "RulL.z",
          "RulR2d.x", "RulR2d.y", "RulL2d.x", "RulL2d.y"]
all = np.concatenate([X, T, N, B, kv.reshape(-1,1), tr.reshape(-1,1),
                 X2d, T2d, N2d, k2d.reshape(-1,1),
                 fa.reshape(-1,1), da.reshape(-1,1),
                 cosa.reshape(-1,1), sina.reshape(-1,1), tana.reshape(-1,1),
                 betaR.reshape(-1,1), betaL.reshape(-1,1),
                 cosbR.reshape(-1,1), cosbL.reshape(-1,1),
                 sinbR.reshape(-1,1), sinbL.reshape(-1,1),
                 RulR, RulL, RulR2d, RulL2d], 1)

path = 'output/result.csv'
with open(path, 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all.tolist())

"""
    polygon
"""

# paper edges
psx = 0
psy = 0
pex = 200
pey = 200

# ruling lengths
RulRlen = Ruling.RulingLength(X2d, RulR2d, psx,psy,pex,pey)
RulLlen = Ruling.RulingLength(X2d, RulL2d, psx,psy,pex,pey)
 
# mat[2][Xcnt+1][4][4], vtx[2][Xcnt+1][4][3], vtx2d[2][Xcnt+1][4][3]
mat, vtx, vtx2d = Polygon.PolyFaces(X, RulR, RulL, X2d, RulR2d, RulL2d, RulRlen, RulLlen)

Polygon.checkPoly(mat, vtx, vtx2d, "output/vtx0.obj", "output/vtx1.obj")
