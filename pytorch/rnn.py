import torch
from torch import nn

input_size = 1
hidden_size = 1
num_layers  = 3

in_x = 1
in_y = 1

rnn = nn.RNN(input_size=input_size,
             hidden_size=hidden_size,
             num_layers = num_layers,
             bias=False)

print('RNN: %s; %s'%(rnn, type(rnn)))

print('Model: %s'%rnn)

for i in range(num_layers):
    exec('rnn.weight_ih_l%i = torch.nn.Parameter(torch.zeros(hidden_size,input_size))'%i)
    exec('rnn.weight_ih_l%i[0][0] = 1'%i)

for l in range(num_layers):
    exec('rnn.weight_hh_l%i = torch.nn.Parameter(torch.zeros(hidden_size,hidden_size))'%l)
    exec('rnn.weight_hh_l%i[0][0] = 1'%l)


print('State_dict,: %s'%rnn.state_dict())

input = torch.zeros(in_x, in_y, input_size)
input[0][0]=1
print('Input: %s'%input)

h0 = torch.randn(num_layers, in_y, hidden_size)
h0 = torch.zeros(num_layers, in_y, hidden_size)
#h0[0][0]=0.5
print('h0: %s'%h0)

output, hn = rnn(input, h0)

print('Output calculated by pyTorch, output: %s'%output)
print('hn: %s'%hn)

print('State_dict,: %s'%rnn.state_dict())

# Export the model
fn = "rnn.onnx"
torch_out = torch.onnx._export(rnn,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)
