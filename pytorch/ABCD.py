import sys
import torch
from torch import nn
import numpy as np

sys.path.append("../onnx")
import ABCD_info as abcd

in_size = 1
out_size = 1

A = nn.Linear(in_size, out_size)
A.weight[0][0] = abcd.A_slope
A.bias[0] = abcd.A_intercept
'''
B = nn.Linear(in_size, out_size)
B.weight[0][0] = 1
B.bias[0] = 0
C = nn.Linear(in_size, out_size)
C.weight[0][0] = 1
C.bias[0] = 0
D = nn.Linear(in_size, out_size)
D.weight[0][0] = 1
D.bias[0] = 0
#'''
B = nn.Sigmoid()

class MyExp(nn.Module):

    def forward(self, input: torch.Tensor):
        return torch.exp(input)

C = MyExp()

m = nn.Sequential(A,B,C)
print('Model: %s'%m)
#print(dir(m))


for i in abcd.test_values:
    input = torch.ones(in_size)*i
    output = m(input)
    print('Output calculated by pytorch (input %s): %s'%(input,output))

# Export the model
fn = "ABCD.onnx"
torch_out = torch.onnx._export(m,             # model being run
                               input,         # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)

def info(a):
    print('Info: %s (%s), %s'%(a.name, a.type, a.shape))

import onnxruntime as rt

sess = rt.InferenceSession(fn)
info(sess.get_inputs()[0])
info(sess.get_outputs()[0])

for i in abcd.test_values:
    
    x = np.array([i],np.float32)

    res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
    print('Output calculated by onnxruntime (input: %s):  %s'%(x,res))


print('Done! ONNX inference')