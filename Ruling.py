# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import numpy as np

def RulingAngle(kv, tr, da, sina):
    kv = kv[3:-3]
    tr = tr[3:-3]
    da = da[3:-3]
    sina = sina[3:-3]
    
    cotbL = (da+tr)/(sina*kv)
    cotbR = (-da+tr)/(sina*kv)
    betaL = np.arctan(1.0/cotbL)
    betaR = np.arctan(1.0/cotbR)
    sinbL = 1.0/np.sqrt(1.0+cotbL*cotbL)
    sinbR = 1.0/np.sqrt(1.0+cotbR*cotbR)
    cosbL = cotbL/np.sqrt(1.0+cotbL*cotbL)
    cosbR = cotbR/np.sqrt(1.0+cotbR*cotbR)
    
    zero = np.array([0.,0.,0.])
    betaR = np.concatenate([zero, betaR, zero])
    betaL = np.concatenate([zero, betaL, zero])
    cosbR = np.concatenate([zero, cosbR, zero])
    cosbL = np.concatenate([zero, cosbL, zero])
    sinbR = np.concatenate([zero, sinbR, zero])
    sinbL = np.concatenate([zero, sinbL, zero])
    
    return betaR, betaL, cosbR, cosbL, sinbR, sinbL


def Ruling(T,N,B,T2d,cosa,sina,cosbR,cosbL,sinbR,sinbL):
    T = T[3:-3]
    N = N[3:-3]
    B = B[3:-3]
    T2d = T2d[3:-3].reshape(-1,2,1)
    cosa = cosa[3:-3].reshape([-1,1])
    sina = sina[3:-3].reshape([-1,1])
    cosbL = cosbL[3:-3].reshape([-1,1])
    cosbR = cosbR[3:-3].reshape([-1,1])
    sinbL = sinbL[3:-3].reshape([-1,1])
    sinbR = sinbR[3:-3].reshape([-1,1])
    
    RulL = cosbL*T +sinbL*cosa*N +sinbL*sina*B
    RulR = cosbR*T -sinbR*cosa*N +sinbR*sina*B
    RulL = RulL / np.linalg.norm(RulL, 2, 1).reshape(-1,1)
    RulR = RulR / np.linalg.norm(RulR, 2, 1).reshape(-1,1)

    matL2d = np.stack([cosbL,-sinbL,sinbL,cosbL], 1).reshape(-1,2,2)
    matR2d = np.stack([cosbR,sinbR,-sinbR,cosbR], 1).reshape(-1,2,2)
    RulL2d = np.matmul(matL2d,T2d).reshape(-1,2)
    RulR2d = np.matmul(matR2d,T2d).reshape(-1,2)

    zero3 = np.zeros(9).reshape(3,3)
    RulR = np.concatenate([zero3, RulR, zero3])
    RulL = np.concatenate([zero3, RulL, zero3])
    zero2 = np.zeros(6).reshape(3,2)
    RulR2d = np.concatenate([zero2, RulR2d, zero2])
    RulL2d = np.concatenate([zero2, RulL2d, zero2])
    
    return RulR, RulL, RulR2d, RulL2d


def RulingLength(X2d, Rul, psx,psy,pex,pey):
    pedge = np.array([[psx,psy],[pex,pey]])
    RulLen = ((pedge - X2d[3:-3].reshape(-1,1,2))/Rul[3:-3].reshape(-1,1,2)).reshape(-1,4)
    RulLen[RulLen < 0] = (pex-psx) + (pey-psy) #対角線より大きい長さ
    RulLen = np.min(RulLen,1)
    zero = np.zeros(3)
    RulLen = np.concatenate([zero, RulLen, zero])
    
    return RulLen
