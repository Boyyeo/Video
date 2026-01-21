

import torch 



## module choleski
''' L = choleski(a)
Choleski decomposition: [L][L]transpose = [a]
x = choleskiSol(L,b)
Solution phase of Choleski's decomposition method
'''
import numpy as np
import math
import torch.nn as nn

class CholesKeyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
      
    def forward(self, lhs, rhs):
        L = choleski(lhs)
        y = choleskiSol(rhs.squeeze(), L.squeeze())
        return y


'''
def choleski(a):
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = torch.sqrt(a[k,k] - torch.dot(a[k,0:k],a[k,0:k]))
        except ValueError:
            print('Matrix is not positive definite')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - torch.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): a[0:k,k] = 0.0
    return a
'''


def choleski(a):
    n = len(a)
    L_list = []
    for i in range(n):
        L_list.append(a[i].clone().squeeze())

    for k in range(n):
        #L_sub = torch.zeros(n)

        try:
            #L_list[k][k] = torch.sqrt(L_list[k][k] - torch.dot(L_list[k][0:k],L_list[k][0:k]))
            L_list[k][k] = torch.sqrt(L_list[k][k] - L_list[k][0:k] @ L_list[k][0:k])
        except ValueError:  
            print('Matrix is not positive definite')
        for i in range(k+1,n):
            #L_list[i][k] = (L_list[i][k] - torch.dot(L_list[i][0:k],L_list[k][0:k]))/L_list[k][k]
            L_list[i][k] = (L_list[i][k] - L_list[i][0:k] @ L_list[k][0:k])/L_list[k][k]
        
    for k in range(n):
        L_list[k][:k] = torch.tensor([0.0])   
    
    L_return = torch.stack(L_list).T
    return L_return


def choleskiSol(b,L):
    print('L shape:{} b shape:{}'.format(L.shape, b.shape))
    n = len(b)
    # Solution of [L]{y} = {b}
    for k in range(n):
        b[k] = (b[k] - torch.dot(L[k,0:k],b[0:k]))/L[k,k]
    # Solution of [L_transpose]{x} = {y}
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - torch.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    return b


ckpt = torch.load('/nfs/home/boyyeo/mediatek/alpamayo/src/alpamayo_r1/debug_dxy_theta_to_v_without_v0.pt')
lhs = ckpt['lhs'].cpu()
rhs = ckpt['rhs'].cpu()
L = ckpt['L'].cpu()       
y = ckpt['y'].cpu()

print('lhs shape:{} rhs shape:{} L shape:{} y shape:{}'.format(lhs.shape, rhs.shape, L.shape, y.shape))



print('lhs:', lhs.shape)
L_recomputed = choleski(lhs.clone().squeeze()) 
print('L and L_recomputed close:', torch.allclose(L, L_recomputed))  
print('difference between L and L_recomputed:', (L - L_recomputed).abs().max())   
print('L:{}\n L_computed:{}'.format(L, L_recomputed))

print('L shape:', L.shape)
print('L_recomputed shape:', L_recomputed.shape)
print('rhs shape:', rhs.shape)
y_recomputed = choleskiSol(rhs.squeeze(), L.squeeze())
print('y and y_recomputed close:', torch.allclose(y, y_recomputed))
print('difference between y and y_recomputed:', (y - y_recomputed).abs().max())




#model = CholesKeyModel()    
#dummy_input = (lhs.clone().squeeze(), rhs.squeeze())
#model_trace = torch.jit.trace(model, dummy_input) 
#model_name = 'choleski_model'
# 跟踪法与直接 torch.onnx.export(model, ...)等价 
#torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx') 