import torch
from torch import nn

input_size = 3
hidden_size = 2
num_layers  = 2

in_x = 2
in_y = 3

rnn = nn.RNN(input_size=input_size, 
             hidden_size=hidden_size,
             num_layers = num_layers)
        
print('Model: %s'%rnn)

print('State_dict,: %s'%rnn.state_dict())

input = torch.randn(in_x, in_y, input_size)
print('Input: %s'%input)

h0 = torch.randn(num_layers, in_y, hidden_size)
output, hn = rnn(input, h0)

print('Output: %s'%output)

# Export the model
fn = "simple.onnx"
torch_out = torch.onnx._export(rnn,             # model being run
                               input,                       # model input (or a tuple for multiple inputs)
                               fn,       # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file

print('Done! Exported to: %s'%fn)