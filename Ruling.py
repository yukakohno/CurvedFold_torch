# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import torch

def RulingAngle(kv, tr, da, sina):
    kv = kv[3:-3]
    tr = tr[3:-3]
    da = da[3:-3]
    sina = sina[3:-3]
    
    cotbL = (da+tr)/(sina*kv)
    cotbR = (-da+tr)/(sina*kv)
    betaL = torch.atan(1.0/cotbL)
    betaR = torch.atan(1.0/cotbR)
    sinbL = 1.0/torch.sqrt(1.0+cotbL*cotbL)
    sinbR = 1.0/torch.sqrt(1.0+cotbR*cotbR)
    cosbL = cotbL/torch.sqrt(1.0+cotbL*cotbL)
    cosbR = cotbR/torch.sqrt(1.0+cotbR*cotbR)
    
    zero = torch.tensor([0.,0.,0.])
    betaR = torch.cat([zero, betaR, zero])
    betaL = torch.cat([zero, betaL, zero])
    cosbR = torch.cat([zero, cosbR, zero])
    cosbL = torch.cat([zero, cosbL, zero])
    sinbR = torch.cat([zero, sinbR, zero])
    sinbL = torch.cat([zero, sinbL, zero])
    
    return betaR, betaL, cosbR, cosbL, sinbR, sinbL


def Ruling(T,N,B,T2d,cosa,sina,cosbR,cosbL,sinbR,sinbL):
    T = T[3:-3]
    N = N[3:-3]
    B = B[3:-3]
    T2d = T2d[3:-3].view(-1,2,1)
    cosa = cosa[3:-3].view([-1,1])
    sina = sina[3:-3].view([-1,1])
    cosbL = cosbL[3:-3].view([-1,1])
    cosbR = cosbR[3:-3].view([-1,1])
    sinbL = sinbL[3:-3].view([-1,1])
    sinbR = sinbR[3:-3].view([-1,1])
    
    RulL = cosbL*T +sinbL*cosa*N +sinbL*sina*B
    RulR = cosbR*T -sinbR*cosa*N +sinbR*sina*B
    RulL = RulL / torch.norm(RulL, 2, dim=1).view(-1,1)
    RulR = RulR / torch.norm(RulR, 2, dim=1).view(-1,1)

    matL2d = torch.stack([cosbL,-sinbL,sinbL,cosbL], dim=1).view(-1,2,2)
    matR2d = torch.stack([cosbR,sinbR,-sinbR,cosbR], dim=1).view(-1,2,2)
    RulL2d = torch.matmul(matL2d,T2d).view(-1,2)
    RulR2d = torch.matmul(matR2d,T2d).view(-1,2)

    zero3 = torch.zeros(3,3)
    RulR = torch.cat([zero3, RulR, zero3])
    RulL = torch.cat([zero3, RulL, zero3])
    zero2 = torch.zeros(3,2)
    RulR2d = torch.cat([zero2, RulR2d, zero2])
    RulL2d = torch.cat([zero2, RulL2d, zero2])
    
    return RulR, RulL, RulR2d, RulL2d


def RulingLength(X2d, Rul, psx,psy,pex,pey):
    pedge = torch.tensor([[psx,psy],[pex,pey]])
    RulLen = ((pedge - X2d[3:-3].view(-1,1,2))/Rul[3:-3].view(-1,1,2)).view(-1,4)
    RulLen[RulLen < 0] = (pex-psx) + (pey-psy) #対角線より大きい長さ
    RulLen = torch.min(RulLen,axis=1)[0]
    zero = torch.zeros(3)
    RulLen = torch.cat([zero, RulLen, zero])
    
    return RulLen
