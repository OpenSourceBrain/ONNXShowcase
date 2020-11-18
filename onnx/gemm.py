import onnx
import numpy as np

node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    alpha=0.25,
    beta=0.35,
    transA=1,
    transB=1
)
a = np.random.ranf([4, 3]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.random.ranf([1, 5]).astype(np.float32)

print('Node: %s'%node)
print(dir(node))
inputs=[a, b, c]
outputs=[a]

from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])

graph = onnx.helper.make_graph(
        nodes=[node],
        name='MyGemm',
        inputs=[X],
        outputs=[X])
#kwargs[str('producer_name')] = 'pgtest'
model = onnx.helper.make_model(graph)
    
'''
y = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_all_attributes')'''