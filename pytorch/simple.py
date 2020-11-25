import torch
from torch import nn

in_size = 1
out_size = 3

m = nn.Linear(in_size, out_size)
m.weight[0][0] = 1
m.weight[1][0] = 2
m.weight[2][0] = 3
for i in range(out_size): m.bias[i] = 0.1
print('Model: %s'%m)
print('Weight: %s'%m.weight)
print('Bias: %s'%m.bias)
print('State_dict,: %s'%m.state_dict())
#print(dir(m))

input = torch.ones(in_size, in_size)
print('Input: %s'%input)

output = m(input)
print('Output: %s'%output)

# Export the model
fn = "simple.onnx"
torch_out = torch.onnx._export(m,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)

