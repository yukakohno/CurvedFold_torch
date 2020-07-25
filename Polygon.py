# -*- coding: utf-8 -*-
"""
@author: ykohno
"""

import torch

def getMat( V0, V1 ):
    
    mat = torch.eye(4)
    rmat = torch.eye(4)
    tmat0 = torch.eye(4)
    tmat1 = torch.eye(4)
    
    if V0.size()[0]<3 or V1.size()[0]<3 or not V0.size()[0] == V1.size()[0]:
        return mat
        
  	# translation
    ave0 = torch.mean(V0, axis=0)
    ave1 = torch.mean(V1, axis=0)
    V0 = V0 - ave0
    V1 = V1 - ave1
    tmat0[:3,3] = -ave0
    tmat1[:3,3] = ave1

    # rotation
    w=torch.tensor(0.0)
    cw=torch.cos(w)
    sw=torch.sin(w)
    p=torch.tensor(0.0)
    cp=torch.cos(p)
    sp=torch.sin(p)
    k=torch.tensor(0.0)
    ck=torch.cos(k)
    sk=torch.sin(k)
    
    for cnt in range(100):
        
        rmat3 = torch.tensor([[cp*ck, -cp*sk, sp],
                             [cw*sk+sw*sp*ck, cw*ck-sw*sp*sk, -sw*cp],
                             [sw*sk-cw*sp*ck, sw*ck+cw*sp*sk, cw*cp]])

        dfdp_coef = torch.tensor([[[0., 0., 0.],
                                   [-sp*ck, sp*sk, cp],
                                   [-cp*sk, -cp*ck, 0.]],
                                  [[-sw*sk+cw*sp*ck, -sw*ck-cw*sp*sk, -cw*cp],
                                   [sw*cp*ck, -sw*cp*sk, sw*sp],
                                   [cw*ck-sw*sp*sk, -cw*sk-sw*sp*ck, 0.]],
                                  [[cw*sk+sw*sp*ck, cw*ck-sw*sp*sk, -sw*cp],
                                   [-cw*cp*ck, cw*cp*sk,-cw*sp],
                                   [sw*ck+cw*sp*sk, -sw*sk+cw*sp*ck, 0.]]])
    
        # F = rmat3 * V0 - V1
        F = torch.matmul(rmat3.view(-1,3,3).expand(4,-1,-1),V0.view(-1,3,1)).squeeze() - V1
        # dfdp = dfdp_coef * V0
        dfdp = torch.matmul(dfdp_coef.view(-1,3,3,3).expand(4,-1,-1,-1),
                            V0.view(-1,1,3,1).expand(-1,3,-1,-1)).squeeze()

        m = torch.stack([
                dfdp[:,0,0] * dfdp[:,0,0] + dfdp[:,1,0] * dfdp[:,1,0] + dfdp[:,2,0] * dfdp[:,2,0],
                dfdp[:,0,0] * dfdp[:,0,1] + dfdp[:,1,0] * dfdp[:,1,1] + dfdp[:,2,0] * dfdp[:,2,1],
                dfdp[:,0,0] * dfdp[:,0,2] + dfdp[:,1,0] * dfdp[:,1,2] + dfdp[:,2,0] * dfdp[:,2,2],
                dfdp[:,0,1] * dfdp[:,0,0] + dfdp[:,1,1] * dfdp[:,1,0] + dfdp[:,2,1] * dfdp[:,2,0],
                dfdp[:,0,1] * dfdp[:,0,1] + dfdp[:,1,1] * dfdp[:,1,1] + dfdp[:,2,1] * dfdp[:,2,1],
                dfdp[:,0,1] * dfdp[:,0,2] + dfdp[:,1,1] * dfdp[:,1,2] + dfdp[:,2,1] * dfdp[:,2,2],
                dfdp[:,0,2] * dfdp[:,0,0] + dfdp[:,1,2] * dfdp[:,1,0] + dfdp[:,2,2] * dfdp[:,2,0],
                dfdp[:,0,2] * dfdp[:,0,1] + dfdp[:,1,2] * dfdp[:,1,1] + dfdp[:,2,2] * dfdp[:,2,1],
                dfdp[:,0,2] * dfdp[:,0,2] + dfdp[:,1,2] * dfdp[:,1,2] + dfdp[:,2,2] * dfdp[:,2,2],
                ], dim=1).view(-1,3,3)

        m_total = torch.sum(m,dim=0).inverse()

        b = torch.matmul(torch.transpose(dfdp,1,2),F.view(-1,3,1)).squeeze()
        b_total = torch.sum(b,dim=0)
        
        d = torch.matmul(m_total,b_total)
        
        if max(torch.abs(d)) < 0.0001:
            #print(d, ", cnt=" , cnt)
            break

        w = w - d[0]
        cw=torch.cos(w)
        sw=torch.sin(w)
        p = p - d[1]
        cp=torch.cos(p)
        sp=torch.sin(p)
        k = k - d[2]
        ck=torch.cos(k)
        sk=torch.sin(k)
    
    rmat[:3,:3] = rmat3
    mat = torch.chain_matmul(tmat1,rmat,tmat0)

    return mat

# mat[2][Xcnt+1][4][4], vtx[2][Xcnt+1][4][3], vtx2d[2][Xcnt+1][4][2]
def PolyFaces(X, RulR, RulL, X2d, RulR2d, RulL2d, RulRlen, RulLlen):
    Xcnt = X.size()[0]
    mat = torch.zeros(2, Xcnt+1, 4, 4)
    vtx = torch.zeros(2, Xcnt+1, 4, 3)
    vtx2d = torch.zeros(2, Xcnt+1, 4, 2)
    
    edgeR = X + RulR * RulRlen.view(-1,1)
    edgeL = X + RulL * RulLlen.view(-1,1)
    edgeR2d = X2d + RulR2d * RulRlen.view(-1,1)
    edgeL2d = X2d + RulL2d * RulLlen.view(-1,1)

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
    for i in range(1,mat[0].size()[0]-1):
        mat[0][i] = getMat(torch.cat((vtx2d[0][i],torch.zeros(4).view(-1,1)), dim=1), vtx[0][i])
        mat[1][i] = getMat(torch.cat((vtx2d[1][i],torch.zeros(4).view(-1,1)), dim=1), vtx[1][i])

    vtx2d[0,0,0]
    vtx2d[0,0,3]
    vtx2d[0,-1,1]
    vtx2d[0,-1,2]
    
    vtx2d[1,0,1]
    vtx2d[1,0,2]
    vtx2d[1,-1,0]
    vtx2d[1,-1,3]
    
    return mat, vtx.detach(), vtx2d.detach()

def checkPoly(mat, vtx, vtx2d, path0, path1):
    Fcnt0 = mat.size()[0]
    Fcnt1 = mat.size()[1]
    Vcnt = vtx.size()[2]
    mat_ = mat.detach()
    vtx2d_ = torch.cat([vtx2d.detach(), torch.tensor([0.,1.]).view(1,1,1,2).expand(Fcnt0,Fcnt1,Vcnt,-1)], dim=3)
    vtx_ = torch.cat([vtx.detach(), torch.tensor([1.]).view(1,1,1,1).expand(Fcnt0,Fcnt1,Vcnt,-1)], dim=3)
    vtx_mat = torch.zeros(vtx_.size())

    for i0 in range(Fcnt0):
        for i1 in range(1,Fcnt1-1):
            for j in range(Vcnt):
                vtx_mat[i0][i1][j] = torch.matmul(mat_[i0][i1], vtx2d_[i0][i1][j])
                
    diff = vtx_ - vtx_mat
    print( 'vertex error:', torch.max(diff[:,1:-1]).item())
    
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
