# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import torch

# k2d[Xcnt] -> X2d,T2d,N2d[Xcnt][2]
def ReconstructX2d( k2d, m2, dX ):
    Xlen = k2d.size()[0]
    T2d = torch.zeros(Xlen, 2)
    N2d = torch.zeros(Xlen, 2)
    X2d = torch.zeros(Xlen, 2)

    T2d[0]=m2[0][:2]
    N2d[0]=m2[1][:2]
    X2d[0]=m2[2][:2]
    
    t = T2d[0] + k2d[0]*dX*N2d[0]
    T2d[1] = t / torch.norm(t, 2) # normalize
    N2d[1] = torch.tensor([-T2d[1][1], T2d[1][0]])

    for i in range(2,Xlen):
        t = T2d[i-2] + 2.0*k2d[i-1]*dX*N2d[i-1]
        T2d[i] = t / torch.norm(t, 2) # normalize
        N2d[i] = torch.tensor([-T2d[i][1], T2d[i][0]])

    for i in range(1,Xlen):
        Tave = T2d[i-1] + T2d[i]
        Tave = Tave / torch.norm(Tave, 2) # normalize
        X2d[i] = X2d[i-1] + Tave*dX

    return X2d, T2d, N2d

def ReconstructX( kv, tr, m3, dX ):
    Xlen = kv.size()[0]
    T = torch.zeros(Xlen, 3)
    N = torch.zeros(Xlen, 3)
    B = torch.zeros(Xlen, 3)
    X = torch.zeros(Xlen, 3)
    
    T[0]=m3[0][:3]
    N[0]=m3[1][:3]
    B[0]=m3[2][:3]
    X[0]=m3[3][:3]

    t = T[0] + kv[0]*dX*N[0]
    T[1] = t / torch.norm(t, 2) # normalize
    n = N[0] -kv[0]*dX*T[0] +tr[0]*dX*B[0]
    N[1] = n / torch.norm(n, 2) # normalize
    b = torch.cross(T[1], N[1])
    B[1] = b / torch.norm(b, 2) # normalize

    for i in range(2,Xlen):
        t = T[i-2] + 2*kv[i-1]*dX*N[i-1]
        T[i] = t / torch.norm(t, 2) # normalize
        n = N[i-2] -2*kv[i-1]*dX*T[i-1] +2*tr[i-1]*dX*B[i-1]
        N[i] = n / torch.norm(n, 2) # normalize
        b = torch.cross(T[i], N[i])
        B[i] = b / torch.norm(b, 2) # normalize

    for i in range(1,Xlen):
        Tave = T[i-1] + T[i]
        Tave = Tave / torch.norm(Tave, 2) # normalize
        X[i] = X[i-1] + Tave*dX

    return X, T, N, B
