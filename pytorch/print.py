import onnx
import sys

filename = "squeezenet.onnx"
if len(sys.argv)==2:
    filename = sys.argv[1]
    
print('Loading ONNX model %s'%filename)
# Load the ONNX model
model = onnx.load(filename)

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
