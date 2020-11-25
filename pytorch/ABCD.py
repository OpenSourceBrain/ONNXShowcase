import torch
from torch import nn

in_size = 1
out_size = 1

A = nn.Linear(in_size, out_size)
A.weight[0][0] = 2
A.bias[0] = 2
B = nn.Linear(in_size, out_size)
B.weight[0][0] = 1
B.bias[0] = 0
C = nn.Linear(in_size, out_size)
C.weight[0][0] = 1
C.bias[0] = 0
D = nn.Linear(in_size, out_size)
D.weight[0][0] = 1
D.bias[0] = 0
#B = nn.Sigmoid()
m = nn.Sequential(A,B,C,D)
print('Model: %s'%m)
#print(dir(m))

input = torch.zeros(in_size, in_size)
print('Input: %s'%input)

output = m(input)
print('Output calculated by pytorch: %s'%output)

# Export the model
fn = "ABCD.onnx"
torch_out = torch.onnx._export(m,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)

def info(a):
    print('Info: %s (%s), %s'%(a.name, a.type, a.shape))

import onnxruntime as rt

sess = rt.InferenceSession(fn)
info(sess.get_inputs()[0])
info(sess.get_outputs()[0])


import numpy
x = numpy.array(input)
res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
print('Output calculated by onnxruntime:  %s'%res)


print('Done! ONNX inference')