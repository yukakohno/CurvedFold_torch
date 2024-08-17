# -*- coding: utf-8 -*-
"""
@author: ykohno
"""
import numpy as np

def get_coeff(Xx, num):
    Xlen = len(Xx)
    jj = np.floor(Xx).astype(np.int32) #/ 小数点以下切捨て
    jj = np.maximum(jj, np.zeros_like(jj))
    jj = np.minimum(jj, np.zeros_like(jj)+num-1)
    dt = Xx - jj.astype(np.float32)
    
    spmat = np.zeros([Xlen,num])
    for i in range(Xlen):
        spmat[i,jj[i]]=1

    return spmat, dt


def interpolate_linear_7(P_val, spmat, dt):
    
    Xlen = len(P_val)
    num = 6

    a = P_val
    b = np.zeros(Xlen)

    for i in range(num):
        b[i] = a[i+1]-a[i]

    abcd = np.concatenate([a.reshape(-1,1),b.reshape(-1,1)],1)[:num]
    coef = np.matmul(spmat,abcd)
    #a+b*x
    X_val = coef[:,0] + coef[:,1]*dt
    return X_val

# http://www5d.biglobe.ne.jp/stssk/maze/spline.html
def interpolate_spline_7(P_val, spmat, dt):

    Xlen = len(P_val)
    num = 6
    
    # ３次多項式の0次係数(a)を設定
    a = P_val
    b = np.zeros(Xlen)
    c = np.zeros(Xlen)
    d = np.zeros(Xlen)
    w = np.zeros(Xlen)

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

    abcd = np.concatenate([a.reshape(-1,1),b.reshape(-1,1),c.reshape(-1,1),d.reshape(-1,1)],1)[:num]
    coef = np.matmul(spmat,abcd)
    X_val = coef[:,0] + (coef[:,1] + (coef[:,2] + coef[:,3] *dt) *dt) *dt
    return X_val
