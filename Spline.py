# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import torch

def get_coeff(Xx, num):
    Xlen = len(Xx)
    jj = torch.floor(Xx).int() #/ 小数点以下切捨て
    jj = torch.max(jj, torch.zeros_like(jj))
    jj = torch.min(jj, torch.zeros_like(jj)+num-1)
    dt = Xx - jj.float()
    
    spmat = torch.zeros([Xlen,num])
    for i in range(Xlen):
        spmat[i,jj[i]]=1

    return spmat, dt


def interpolate_linear_7(P_val, spmat, dt):
    
    Xlen = len(P_val)
    num = 6

    a = P_val
    b = torch.zeros(Xlen)

    for i in range(num):
        b[i] = a[i+1]-a[i]

    abcd = torch.cat([a.view(-1,1),b.view(-1,1)],dim=1)[:num]
    coef = torch.matmul(spmat,abcd)
    #a+b*x
    X_val = coef[:,0] + coef[:,1]*dt
    return X_val

# http://www5d.biglobe.ne.jp/stssk/maze/spline.html
def interpolate_spline_7(P_val, spmat, dt):

    Xlen = len(P_val)
    num = 6
    
    # ３次多項式の0次係数(a)を設定
    a = P_val
    b = torch.zeros(Xlen)
    c = torch.zeros(Xlen)
    d = torch.zeros(Xlen)
    w = torch.zeros(Xlen)

    # ３次多項式の2次係数(c)を計算
    # 連立方程式を解く。
    # 但し、一般解法でなくスプライン計算にチューニングした方法
    c[0] = 0.0
    for i in range(1, num):
        c[i] = 3.0 * (a[i-1] - 2.0 * a[i] + a[i+1])

    # 左下を消す
    for i in range(1, num):
        tmp = 4.0 - w[i-1]
        c[i] = (c[i] - c[i-1])/tmp
        w[i] = 1.0 / tmp
        
    # 左下を消す
    for i in reversed(range(0,num-1)):
        c[i] = c[i] - c[i+1] * w[i]

    # ３次多項式の1次係数(b)と3次係数(b)を計算
    for i in range(num):
        d[i] = ( c[i+1] - c[i]) / 3.0
        b[i] = a[i+1] - a[i] - c[i] - d[i]

    abcd = torch.cat([a.view(-1,1),b.view(-1,1),c.view(-1,1),d.view(-1,1)],dim=1)[:num]
    coef = torch.matmul(spmat,abcd)
    X_val = coef[:,0] + (coef[:,1] + (coef[:,2] + coef[:,3] *dt) *dt) *dt
    return X_val
