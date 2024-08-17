# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import numpy as np

# k2d[Xcnt] -> X2d,T2d,N2d[Xcnt][2]
def ReconstructX2d( k2d, m2, dX ):
    Xlen = k2d.shape[0]
    T2d = np.zeros(Xlen*2).reshape(Xlen,2)
    N2d = np.zeros(Xlen*2).reshape(Xlen,2)
    X2d = np.zeros(Xlen*2).reshape(Xlen,2)

    T2d[0]=m2[0][:2]
    N2d[0]=m2[1][:2]
    X2d[0]=m2[2][:2]
    
    t = T2d[0] + k2d[0]*dX*N2d[0]
    T2d[1] = t / np.linalg.norm(t, 2) # normalize
    N2d[1] = np.array([-T2d[1][1], T2d[1][0]])

    for i in range(2,Xlen):
        t = T2d[i-2] + 2.0*k2d[i-1]*dX*N2d[i-1]
        T2d[i] = t / np.linalg.norm(t, 2) # normalize
        N2d[i] = np.array([-T2d[i][1], T2d[i][0]])

    for i in range(1,Xlen):
        Tave = T2d[i-1] + T2d[i]
        Tave = Tave / np.linalg.norm(Tave, 2) # normalize
        X2d[i] = X2d[i-1] + Tave*dX

    return X2d, T2d, N2d

def ReconstructX( kv, tr, m3, dX ):
    Xlen = kv.shape[0]
    T = np.zeros(Xlen*3).reshape(Xlen,3)
    N = np.zeros(Xlen*3).reshape(Xlen,3)
    B = np.zeros(Xlen*3).reshape(Xlen,3)
    X = np.zeros(Xlen*3).reshape(Xlen,3)
    
    T[0]=m3[0][:3]
    N[0]=m3[1][:3]
    B[0]=m3[2][:3]
    X[0]=m3[3][:3]

    t = T[0] + kv[0]*dX*N[0]
    T[1] = t / np.linalg.norm(t, 2) # normalize
    n = N[0] -kv[0]*dX*T[0] +tr[0]*dX*B[0]
    N[1] = n / np.linalg.norm(n, 2) # normalize
    b = np.cross(T[1], N[1])
    B[1] = b / np.linalg.norm(b, 2) # normalize

    for i in range(2,Xlen):
        t = T[i-2] + 2*kv[i-1]*dX*N[i-1]
        T[i] = t / np.linalg.norm(t, 2) # normalize
        n = N[i-2] -2*kv[i-1]*dX*T[i-1] +2*tr[i-1]*dX*B[i-1]
        N[i] = n / np.linalg.norm(n, 2) # normalize
        b = np.cross(T[i], N[i])
        B[i] = b / np.linalg.norm(b, 2) # normalize

    for i in range(1,Xlen):
        Tave = T[i-1] + T[i]
        Tave = Tave / np.linalg.norm(Tave, 2) # normalize
        X[i] = X[i-1] + Tave*dX

    return X, T, N, B


# kv,tr[Xcnt] -> X,T,N,B[Xcnt][3]
def ReconstructX_grad( kv, tr, m3, dX ):
    Xlen = kv.shape[0]
    T=m3[0][:3]
    N=m3[1][:3]
    B=m3[2][:3]
    X=m3[3][:3]

    t = T + kv[0]*dX*N[0]
    t = t / np.linalg.norm(t, 2) # normalize
    T = np.stack([T, t])   
    n = N -kv[0]*dX*T[0] +tr[0]*dX*B[0]
    n = n / np.linalg.norm(n, 2) # normalize
    N = np.stack([N, n])
    b = np.cross(t, n)
    b = b / np.linalg.norm(b, 2) # normalize
    B = np.stack([B, b])    

    for i in range(2,Xlen):
        t = T[i-2] + 2*kv[i-1]*dX*N[i-1]
        t = t / np.linalg.norm(t, 2) # normalize
        T = np.concatenate((T, t.reshape(1,-1)), 0)
        n = N[i-2] -2*kv[i-1]*dX*T[i-1] +2*tr[i-1]*dX*B[i-1]
        n = n / np.linalg.norm(n, 2) # normalize
        N = np.concatenate((N, n.reshape(1,-1)), 0)
        b = np.cross(t, n)
        b = b / np.linalg.norm(b, 2) # normalize
        B = np.concatenate((B, b.reshape(1,-1)), 0)

    X = X.reshape(1,-1)
    for i in range(1,Xlen):
        Tave = T[i-1] + T[i]
        Tave = Tave / np.linalg.norm(Tave, 2) # normalize
        x = X[i-1] + Tave*dX
        X = np.concatenate((X, x.reshape(1,-1)), 0)

    return X, T, N, B
