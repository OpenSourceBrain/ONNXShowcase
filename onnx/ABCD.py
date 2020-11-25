import onnx
import numpy as np


A = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    alpha=2,
    beta=2,
    transA=1,
    transB=1
)
a = np.random.ranf([4, 3]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.random.ranf([1, 5]).astype(np.float32)

print('Node: %s'%A)
#print(dir(A))
inputs=[a, b, c]
outputs=[a]
X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [3, 2])

graph = onnx.helper.make_graph(
        nodes=[A],
        name='ABCD',
        inputs=[X],
        outputs=[X])
#kwargs[str('producer_name')] = 'pgtest'
onnx_model = onnx.helper.make_model(graph)

print('Model: %s'%onnx_model)

onnx.save(onnx_model, 'ABCD.onnx')
    
