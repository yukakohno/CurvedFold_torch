# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import numpy as np

def calcDiffAngle(fa, dX):
    Xlen = fa.shape[0]
    da = np.zeros(fa.shape)
    for i in range(1,Xlen-1):
        da[i] = (fa[i+1]-fa[i-1]) *0.5 /dX
    return da

def calcDiffAngle_grad(fa, dX):
    Xlen = fa.shape[0]
    da = np.array([0.])
    for i in range(1,Xlen-1):
        dai = (fa[i+1]-fa[i-1]) *0.5 /dX
        da = np.concatenate((da, dai.reshape(1)))
    da = np.concatenate((da, np.array([0.])))
    return da


def calcTNk2d( X2d, dX ):
    Xlen = X2d.shape[0]
    T2d = np.zeros(X2d.shape)
    N2d = np.zeros(X2d.shape)
    k2d = np.zeros(X2d.shape[0])

    for i in range(1,Xlen-1):
        t = (X2d[i+1]-X2d[i-1]) *0.5 /dX
        T2d[i] = t / np.linalg.norm(t, 2) # normalize

    for i in range(2,Xlen-2):
        n = (T2d[i+1]-T2d[i-1]) *0.5 /dX
        k = np.sign(T2d[i][0]*n[1]-n[0]*T2d[i][1]) * np.linalg.norm(n, 2)
        k2d[i] = k
        N2d[i] = n / k # normalize

    return T2d, N2d, k2d

def calcTNBkt( X, dX ):
    Xlen = X.shape[0]
    T = np.zeros(X.shape)
    N = np.zeros(X.shape)
    B = np.zeros(X.shape)
    kv = np.zeros(X.shape[0])
    tr0 = np.zeros(X.shape[0])
    tr1 = np.zeros(X.shape[0])
    ip0 = np.zeros(X.shape[0])
    ip1 = np.zeros(X.shape[0])
    dT = np.zeros(X.shape)
    dN = np.zeros(X.shape)
    dB = np.zeros(X.shape)

    for i in range(1,Xlen-1):
        t = (X[i+1]-X[i-1]) *0.5 /dX
        T[i] = t / np.linalg.norm(t, 2) # normalize

    for i in range(2,Xlen-2):
        dT[i] = (T[i+1]-T[i-1]) *0.5 /dX
        # original code for sign:
		# if((i==2 && Nx0[i]*Nx0[0] + Ny0[i]*Ny0[0] + Nz0[i]*Nz0[0] < 0)
		#    || (i>2 && Nx0[i]*Nx0[i-1] + Ny0[i]*Ny0[i-1] + Nz0[i]*Nz0[i-1] < 0))
        kv[i] = np.sign(np.cross(T[i],dT[i])[2]) * np.linalg.norm(dT[i], 2)
        N[i] = dT[i] / kv[i] # normalize
        b = np.cross(T[i],N[i])
        B[i] = b / np.linalg.norm(b, 2) # normalize

    for i in range(3,Xlen-3):
        dN[i] = (N[i+1]-N[i-1]) *0.5 /dX
        dB[i] = (B[i+1]-B[i-1]) *0.5 /dX
        
        nkt = dN[i] + kv[i]*T[i]
        tr0[i] = np.linalg.norm(nkt, 2)
        ip0[i] = np.dot(nkt/tr0[i],B[i])
        
        tr1[i] = np.linalg.norm(dB[i], 2)
        ip1[i] = np.dot(-dB[i]/tr1[i], N[i])
               
    return T, N, B, kv, tr0

def calcTNBkt_grad( X, dX ):
    Xlen = X.shape[0]

    T = np.array([[0.,0.,0.]])
    for i in range(1,Xlen-1):
        t = (X[i+1]-X[i-1]) *0.5 /dX
        t = t / np.linalg.norm(t, 2) # normalize
        T = np.concatenate((T, t.reshape(1,-1)), 0)
    T = np.concatenate((T, np.array([[0.,0.,0.]])))

    dT = np.array([[0.,0.,0.],[0.,0.,0.]])
    N = np.array([[0.,0.,0.],[0.,0.,0.]])
    B = np.array([[0.,0.,0.],[0.,0.,0.]])
    kv = np.array([0.,0.])
    for i in range(2,Xlen-2):
        dt = (T[i+1]-T[i-1]) *0.5 /dX
        dT = np.concatenate((dT, dt.reshape(1,-1)), 0)
        # original code for sign:
		# if((i==2 && Nx0[i]*Nx0[0] + Ny0[i]*Ny0[0] + Nz0[i]*Nz0[0] < 0)
		#    || (i>2 && Nx0[i]*Nx0[i-1] + Ny0[i]*Ny0[i-1] + Nz0[i]*Nz0[i-1] < 0))
        kvi = np.sign(np.cross(T[i],dT[i])[2]) * np.linalg.norm(dT[i], 2)
        kv = np.concatenate((kv, kvi.reshape(1)))
        n = dt / kvi # normalize
        N = np.concatenate((N, n.reshape(1,-1)), 0)        
        b = np.cross(T[i],n)
        b = b / np.linalg.norm(b, 2) # normalize
        B = np.concatenate((B, b.reshape(1,-1)), 0)        
    dT = np.concatenate((dT, np.array([[0.,0.,0.],[0.,0.,0.]])))
    N = np.concatenate((N, np.array([[0.,0.,0.],[0.,0.,0.]])))
    B = np.concatenate((B, np.array([[0.,0.,0.],[0.,0.,0.]])))
    kv = np.concatenate((kv, np.array([0.,0.])))

    dN = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    dB = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    tr0 = np.array([0.,0.,0.])
    tr1 = np.array([0.,0.,0.])
    ip0 = np.array([0.,0.,0.])
    ip1 = np.array([0.,0.,0.])
    for i in range(3,Xlen-3):
        dn = (N[i+1]-N[i-1]) *0.5 /dX
        dN = np.concatenate((dN, dn.reshape(1,-1)), 0)
        db = (B[i+1]-B[i-1]) *0.5 /dX
        dB = np.concatenate((dB, db.reshape(1,-1)), 0)
        nkt = dn + kv[i]*T[i]
        tr0i = np.linalg.norm(nkt, 2)
        tr0 = np.concatenate((tr0, tr0i.reshape(1)))
        ip0i = np.dot(nkt/tr0i,B[i])
        ip0 = np.concatenate((ip0, ip0i.reshape(1)))        
        tr1i = np.linalg.norm(db, 2)
        tr1 = np.concatenate((tr1, tr1i.reshape(1)))
        ip1i = np.dot(-db/tr1i, N[i])
        ip1 = np.concatenate((ip1, ip1i.reshape(1)))
    dN = np.concatenate((dN, np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])))
    dB = np.concatenate((dB, np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])))
    tr0 = np.concatenate((tr0, np.array([0.,0.,0.])))
    tr1 = np.concatenate((tr1, np.array([0.,0.,0.])))
    ip0 = np.concatenate((ip0, np.array([0.,0.,0.])))
    ip1 = np.concatenate((ip1, np.array([0.,0.,0.])))
               
    return T, N, B, kv, tr0
