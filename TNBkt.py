# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import torch

def calcDiffAngle(fa, dX):
    Xlen = fa.size()[0]
    da = torch.zeros(fa.size())
    for i in range(1,Xlen-1):
        da[i] = (fa[i+1]-fa[i-1]) *0.5 /dX
    return da

def calcDiffAngle_grad(fa, dX):
    Xlen = fa.size()[0]
    da = torch.tensor([0.])
    for i in range(1,Xlen-1):
        dai = (fa[i+1]-fa[i-1]) *0.5 /dX
        da = torch.cat((da, dai.view(1)))
    da = torch.cat((da, torch.tensor([0.])))
    return da


def calcTNk2d( X2d, dX ):
    Xlen = X2d.size(0)
    T2d = torch.zeros(X2d.size())
    N2d = torch.zeros(X2d.size())
    k2d = torch.zeros(X2d.size(0))

    for i in range(1,Xlen-1):
        t = (X2d[i+1]-X2d[i-1]) *0.5 /dX
        T2d[i] = t / torch.norm(t, 2) # normalize

    for i in range(2,Xlen-2):
        n = (T2d[i+1]-T2d[i-1]) *0.5 /dX
        k = (T2d[i][0]*n[1]-n[0]*T2d[i][1]).sign() * torch.norm(n, 2)
        k2d[i] = k
        N2d[i] = n / k # normalize

    return T2d, N2d, k2d

def calcTNBkt( X, dX ):
    Xlen = X.size(0)
    T = torch.zeros(X.size())
    N = torch.zeros(X.size())
    B = torch.zeros(X.size())
    kv = torch.zeros(X.size(0))
    tr0 = torch.zeros(X.size(0))
    tr1 = torch.zeros(X.size(0))
    ip0 = torch.zeros(X.size(0))
    ip1 = torch.zeros(X.size(0))
    dT = torch.zeros(X.size())
    dN = torch.zeros(X.size())
    dB = torch.zeros(X.size())

    for i in range(1,Xlen-1):
        t = (X[i+1]-X[i-1]) *0.5 /dX
        T[i] = t / torch.norm(t, 2) # normalize

    for i in range(2,Xlen-2):
        dT[i] = (T[i+1]-T[i-1]) *0.5 /dX
        # original code for sign:
		# if((i==2 && Nx0[i]*Nx0[0] + Ny0[i]*Ny0[0] + Nz0[i]*Nz0[0] < 0)
		#    || (i>2 && Nx0[i]*Nx0[i-1] + Ny0[i]*Ny0[i-1] + Nz0[i]*Nz0[i-1] < 0))
        kv[i] = torch.cross(T[i],dT[i])[2].sign() * torch.norm(dT[i], 2)
        N[i] = dT[i] / kv[i] # normalize
        b = torch.cross(T[i],N[i])
        B[i] = b / torch.norm(b, 2) # normalize

    for i in range(3,Xlen-3):
        dN[i] = (N[i+1]-N[i-1]) *0.5 /dX
        dB[i] = (B[i+1]-B[i-1]) *0.5 /dX
        
        nkt = dN[i] + kv[i]*T[i]
        tr0[i] = torch.norm(nkt, 2)
        ip0[i] = torch.dot(nkt/tr0[i],B[i])
        
        tr1[i] = torch.norm(dB[i], 2)
        ip1[i] = torch.dot(-dB[i]/tr1[i], N[i])
               
    return T, N, B, kv, tr0

def calcTNBkt_grad( X, dX ):
    Xlen = X.size(0)

    T = torch.tensor([[0.,0.,0.]])
    for i in range(1,Xlen-1):
        t = (X[i+1]-X[i-1]) *0.5 /dX
        t = t / torch.norm(t, 2) # normalize
        T = torch.cat((T, t.view(1,-1)), dim=0)
    T = torch.cat((T, torch.tensor([[0.,0.,0.]])))

    dT = torch.tensor([[0.,0.,0.],[0.,0.,0.]])
    N = torch.tensor([[0.,0.,0.],[0.,0.,0.]])
    B = torch.tensor([[0.,0.,0.],[0.,0.,0.]])
    kv = torch.tensor([0.,0.])
    for i in range(2,Xlen-2):
        dt = (T[i+1]-T[i-1]) *0.5 /dX
        dT = torch.cat((dT, dt.view(1,-1)), dim=0)
        # original code for sign:
		# if((i==2 && Nx0[i]*Nx0[0] + Ny0[i]*Ny0[0] + Nz0[i]*Nz0[0] < 0)
		#    || (i>2 && Nx0[i]*Nx0[i-1] + Ny0[i]*Ny0[i-1] + Nz0[i]*Nz0[i-1] < 0))
        kvi = torch.cross(T[i],dT[i])[2].sign() * torch.norm(dT[i], 2)
        kv = torch.cat((kv, kvi.view(1)))
        n = dt / kvi # normalize
        N = torch.cat((N, n.view(1,-1)), dim=0)        
        b = torch.cross(T[i],n)
        b = b / torch.norm(b, 2) # normalize
        B = torch.cat((B, b.view(1,-1)), dim=0)        
    dT = torch.cat((dT, torch.tensor([[0.,0.,0.],[0.,0.,0.]])))
    N = torch.cat((N, torch.tensor([[0.,0.,0.],[0.,0.,0.]])))
    B = torch.cat((B, torch.tensor([[0.,0.,0.],[0.,0.,0.]])))
    kv = torch.cat((kv, torch.tensor([0.,0.])))

    dN = torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    dB = torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    tr0 = torch.tensor([0.,0.,0.])
    tr1 = torch.tensor([0.,0.,0.])
    ip0 = torch.tensor([0.,0.,0.])
    ip1 = torch.tensor([0.,0.,0.])
    for i in range(3,Xlen-3):
        dn = (N[i+1]-N[i-1]) *0.5 /dX
        dN = torch.cat((dN, dn.view(1,-1)), dim=0)
        db = (B[i+1]-B[i-1]) *0.5 /dX
        dB = torch.cat((dB, db.view(1,-1)), dim=0)
        nkt = dn + kv[i]*T[i]
        tr0i = torch.norm(nkt, 2)
        tr0 = torch.cat((tr0, tr0i.view(1)))
        ip0i = torch.dot(nkt/tr0i,B[i])
        ip0 = torch.cat((ip0, ip0i.view(1)))        
        tr1i = torch.norm(db, 2)
        tr1 = torch.cat((tr1, tr1i.view(1)))
        ip1i = torch.dot(-db/tr1i, N[i])
        ip1 = torch.cat((ip1, ip1i.view(1)))
    dN = torch.cat((dN, torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])))
    dB = torch.cat((dB, torch.tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])))
    tr0 = torch.cat((tr0, torch.tensor([0.,0.,0.])))
    tr1 = torch.cat((tr1, torch.tensor([0.,0.,0.])))
    ip0 = torch.cat((ip0, torch.tensor([0.,0.,0.])))
    ip1 = torch.cat((ip1, torch.tensor([0.,0.,0.])))
               
    return T, N, B, kv, tr0
