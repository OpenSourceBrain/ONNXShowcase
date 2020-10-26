import torch
from torch import nn

in_size = 1
out_size = 3

m = nn.Linear(in_size, out_size)
m.weight[0][0] = 1
m.bias[0] =0.07
print('Model: %s'%m)
print('Weight: %s'%m.weight)
print('Bias: %s'%m.bias)
print('State_dict,: %s'%m.state_dict())
#print(dir(m))

input = torch.zeros(in_size, in_size)
print('Input: %s'%input)

output = m(input)
print('Output: %s'%output)

# Export the model
fn = "simpler.onnx"
torch_out = torch.onnx._export(m,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)

