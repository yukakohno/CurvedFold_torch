# -*- coding: utf-8 -*-
"""
@author: ykohno
"""

import numpy as np

def getMat( V0, V1 ):
    
    mat = np.eye(4)
    rmat = np.eye(4)
    tmat0 = np.eye(4)
    tmat1 = np.eye(4)
    
    if V0.shape[0]<3 or V1.shape[0]<3 or not V0.shape[0] == V1.shape[0]:
        return mat
        
  	# translation
    ave0 = np.mean(V0, axis=0)
    ave1 = np.mean(V1, axis=0)
    V0 = V0 - ave0
    V1 = V1 - ave1
    tmat0[:3,3] = -ave0
    tmat1[:3,3] = ave1

    # rotation
    w=np.array(0.0)
    cw=np.cos(w)
    sw=np.sin(w)
    p=np.array(0.0)
    cp=np.cos(p)
    sp=np.sin(p)
    k=np.array(0.0)
    ck=np.cos(k)
    sk=np.sin(k)
    
    for cnt in range(100):
        
        rmat3 = np.array([[cp*ck, -cp*sk, sp],
                             [cw*sk+sw*sp*ck, cw*ck-sw*sp*sk, -sw*cp],
                             [sw*sk-cw*sp*ck, sw*ck+cw*sp*sk, cw*cp]])

        dfdp_coef = np.array([[[0., 0., 0.],
                                   [-sp*ck, sp*sk, cp],
                                   [-cp*sk, -cp*ck, 0.]],
                                  [[-sw*sk+cw*sp*ck, -sw*ck-cw*sp*sk, -cw*cp],
                                   [sw*cp*ck, -sw*cp*sk, sw*sp],
                                   [cw*ck-sw*sp*sk, -cw*sk-sw*sp*ck, 0.]],
                                  [[cw*sk+sw*sp*ck, cw*ck-sw*sp*sk, -sw*cp],
                                   [-cw*cp*ck, cw*cp*sk,-cw*sp],
                                   [sw*ck+cw*sp*sk, -sw*sk+cw*sp*ck, 0.]]])
    
        # F = rmat3 * V0 - V1
        F = np.matmul(np.tile(rmat3.reshape(1,3,3),(4,1,1)),V0.reshape(-1,3,1)).squeeze() - V1
        # dfdp = dfdp_coef * V0
        dfdp = np.matmul(np.tile(dfdp_coef.reshape(1,3,3,3),(4,1,1,1)), np.tile(V0.reshape(4,1,3,1),(1,3,1,1))).squeeze()

        m = np.stack([
                dfdp[:,0,0] * dfdp[:,0,0] + dfdp[:,1,0] * dfdp[:,1,0] + dfdp[:,2,0] * dfdp[:,2,0],
                dfdp[:,0,0] * dfdp[:,0,1] + dfdp[:,1,0] * dfdp[:,1,1] + dfdp[:,2,0] * dfdp[:,2,1],
                dfdp[:,0,0] * dfdp[:,0,2] + dfdp[:,1,0] * dfdp[:,1,2] + dfdp[:,2,0] * dfdp[:,2,2],
                dfdp[:,0,1] * dfdp[:,0,0] + dfdp[:,1,1] * dfdp[:,1,0] + dfdp[:,2,1] * dfdp[:,2,0],
                dfdp[:,0,1] * dfdp[:,0,1] + dfdp[:,1,1] * dfdp[:,1,1] + dfdp[:,2,1] * dfdp[:,2,1],
                dfdp[:,0,1] * dfdp[:,0,2] + dfdp[:,1,1] * dfdp[:,1,2] + dfdp[:,2,1] * dfdp[:,2,2],
                dfdp[:,0,2] * dfdp[:,0,0] + dfdp[:,1,2] * dfdp[:,1,0] + dfdp[:,2,2] * dfdp[:,2,0],
                dfdp[:,0,2] * dfdp[:,0,1] + dfdp[:,1,2] * dfdp[:,1,1] + dfdp[:,2,2] * dfdp[:,2,1],
                dfdp[:,0,2] * dfdp[:,0,2] + dfdp[:,1,2] * dfdp[:,1,2] + dfdp[:,2,2] * dfdp[:,2,2],
                ], 1).reshape(-1,3,3)

        m_total = np.linalg.pinv(np.sum(m,0))

        b = np.matmul(np.transpose(dfdp,(0,2,1)),F.reshape(-1,3,1)).squeeze()
        b_total = np.sum(b,0)
        
        d = np.matmul(m_total,b_total)
        
        if max(np.abs(d)) < 0.0001:
            #print(d, ", cnt=" , cnt)
            break

        w = w - d[0]
        cw=np.cos(w)
        sw=np.sin(w)
        p = p - d[1]
        cp=np.cos(p)
        sp=np.sin(p)
        k = k - d[2]
        ck=np.cos(k)
        sk=np.sin(k)
    
    rmat[:3,:3] = rmat3
    #mat = np.chain_matmul(tmat1,rmat,tmat0)
    mat = np.matmul(rmat,tmat0)
    mat = np.matmul(tmat1,mat)

    return mat

# mat[2][Xcnt+1][4][4], vtx[2][Xcnt+1][4][3], vtx2d[2][Xcnt+1][4][2]
def PolyFaces(X, RulR, RulL, X2d, RulR2d, RulL2d, RulRlen, RulLlen):
    Xcnt = X.shape[0]
    mat = np.zeros(2*(Xcnt+1)*4*4).reshape(2, Xcnt+1, 4, 4)
    vtx = np.zeros(2*(Xcnt+1)*4*3).reshape(2, Xcnt+1, 4, 3)
    vtx2d = np.zeros(2*(Xcnt+1)*4*2).reshape(2, Xcnt+1, 4, 2)
    
    edgeR = X + RulR * RulRlen.reshape(-1,1)
    edgeL = X + RulL * RulLlen.reshape(-1,1)
    edgeR2d = X2d + RulR2d * RulRlen.reshape(-1,1)
    edgeL2d = X2d + RulL2d * RulLlen.reshape(-1,1)

    #right    
    vtx2d[0,1:,0] = X2d
    vtx2d[0,:-1,1] = X2d
    vtx2d[0,:-1,2] = edgeR2d
    vtx2d[0,1:,3] = edgeR2d
    vtx[0,1:,0] = X
    vtx[0,:-1,1] = X
    vtx[0,:-1,2] = edgeR
    vtx[0,1:,3] = edgeR
    
    #left
    vtx2d[1,:-1,0] = X2d
    vtx2d[1,1:,1] = X2d
    vtx2d[1,1:,2] = edgeL2d
    vtx2d[1,:-1,3] = edgeL2d
    vtx[1,:-1,0] = X
    vtx[1,1:,1] = X
    vtx[1,1:,2] = edgeL
    vtx[1,:-1,3] = edgeL

    # does anyone know how to do this without for loop?
    for i in range(1,mat[0].shape[0]-1):
        mat[0][i] = getMat(np.concatenate((vtx2d[0][i],np.zeros(4).reshape(-1,1)), 1), vtx[0][i])
        mat[1][i] = getMat(np.concatenate((vtx2d[1][i],np.zeros(4).reshape(-1,1)), 1), vtx[1][i])

    vtx2d[0,0,0]
    vtx2d[0,0,3]
    vtx2d[0,-1,1]
    vtx2d[0,-1,2]
    
    vtx2d[1,0,1]
    vtx2d[1,0,2]
    vtx2d[1,-1,0]
    vtx2d[1,-1,3]
    
    return mat, vtx, vtx2d

def checkPoly(mat, vtx, vtx2d, path0, path1):
    Fcnt0 = mat.shape[0]
    Fcnt1 = mat.shape[1]
    Vcnt = vtx.shape[2]
    mat_ = mat
    vtx2d_ = np.concatenate([vtx2d, np.tile(np.array([0.,1.]).reshape(1,1,1,2),(Fcnt0,Fcnt1,Vcnt,1))], 3)
    vtx_ = np.concatenate([vtx, np.tile(np.array([1.]).reshape(1,1,1,1),(Fcnt0,Fcnt1,Vcnt,1))], 3)
    vtx_mat = np.zeros(vtx_.shape)

    for i0 in range(Fcnt0):
        for i1 in range(1,Fcnt1-1):
            for j in range(Vcnt):
                vtx_mat[i0][i1][j] = np.matmul(mat_[i0][i1], vtx2d_[i0][i1][j])
                
    diff = vtx_ - vtx_mat
    print( 'vertex error:', np.max(diff[:,1:-1]).item())
    
    """    
    for i0 in range(Fcnt0):
        for i1 in range(Fcnt1):
            print(diff[i0][i1])
    """
    
    with open(path0, mode='w') as f:
        idx = 1
        for i0 in range(Fcnt0):
            for i1 in range(1,Fcnt1-1):
                for j in range(Vcnt):
                    s = "v " + str(vtx[i0][i1][j][0].item()) + " " + str(vtx[i0][i1][j][1].item()) + " " + str(vtx[i0][i1][j][2].item()) + "\n"
                    f.write(s)
                s = "f"
                for j in range(Vcnt):
                    s = s + " " + str(idx)
                    idx = idx + 1
                s = s + "\n"
                f.write(s)

    with open(path1, mode='w') as f:
        idx = 1
        for i0 in range(Fcnt0):
            for i1 in range(1,Fcnt1-1):
                for j in range(Vcnt):
                    s = "v " + str(vtx_mat[i0][i1][j][0].item()) + " " + str(vtx_mat[i0][i1][j][1].item()) + " " + str(vtx_mat[i0][i1][j][2].item()) + "\n"
                    f.write(s)
                s = "f"
                for j in range(Vcnt):
                    s = s + " " + str(idx)
                    idx = idx + 1
                s = s + "\n"
                f.write(s)

    return
