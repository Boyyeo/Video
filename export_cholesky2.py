

import torch 



## module choleski
''' L = choleski(a)
Choleski decomposition: [L][L]transpose = [a]
x = choleskiSol(L,b)
Solution phase of Choleski's decomposition method
'''
import mtk_converter
import numpy as np
import math
import torch.nn as nn

class CholesKeyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
      
    def forward(self, lhs):
        L = choleski(lhs)
        return L



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
        
        L_list[k][:k] = torch.tensor([0.0])   

    return L


    #for k in range(1,n): a[0:k,k] = 0.0
    #return a

def choleskiSol(b,L):
    n = len(b)
    # Solution of [L]{y} = {b}
    for k in range(n):
        #b[k] = (b[k] - torch.dot(L[k,0:k],b[0:k]))/L[k,k]
        b[k] = (b[k] - L[k,0:k] @ b[0:k])/L[k,k]
    # Solution of [L_transpose]{x} = {y}
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - L[k+1:n,k] @ b[k+1:n])/L[k,k]
    return b


ckpt = torch.load('/nfs/home/boyyeo/mediatek/alpamayo/src/alpamayo_r1/debug_dxy_theta_to_v_without_v0.pt')
lhs = ckpt['lhs'].detach().cpu()
rhs = ckpt['rhs'].detach().cpu()
L = ckpt['L'].detach().cpu()       
y = ckpt['y'].detach().cpu()




model = CholesKeyModel()    
dummy_input = (lhs.clone().squeeze())
model_trace = torch.jit.trace(model, dummy_input) 
torch.jit.save(model_trace, 'choleski_model_trace.pt')


#model_name = 'choleski_model'
# 跟踪法与直接 torch.onnx.export(model, ...)等价 
#torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx') 
converter = mtk_converter.PyTorchConverter.from_script_module_file('choleski_model_trace.pt', [[16,16]])
_ = converter.convert_to_tflite(output_file='choleski_model_trace.tflite')
