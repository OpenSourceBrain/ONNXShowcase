import onnx
import numpy as np

import ABCD_info as abcd

A = onnx.helper.make_node(
    'Sigmoid',
    name='A',
    inputs=['input0'],
    outputs=['A_out'],
    )
B = onnx.helper.make_node(
    'Sigmoid',
    name='B',
    inputs=['A_out'],
    outputs=['B_out'],
    )
C = onnx.helper.make_node(
    'Sigmoid',
    name='C',
    inputs=['A_out'],
    outputs=['C_out'],
    )
D = onnx.helper.make_node(
    'Add',
    name='D',
    inputs=['B_out','C_out'],
    outputs=['D_out'],
    )
'''
    alpha=abcd.A_slope,
    beta=abcd.A_intercept,
    transA=1,
    transB=1
)
'''
print('Node: %s'%A)


input0 = onnx.helper.make_tensor_value_info('input0', onnx.TensorProto.FLOAT, [1])
output0 = onnx.helper.make_tensor_value_info('D_out', onnx.TensorProto.FLOAT, [1])

graph = onnx.helper.make_graph(
        nodes=[A,B,C,D],
        name='ABCD',
        inputs=[input0],
        outputs=[output0])
#kwargs[str('producer_name')] = 'pgtest'
onnx_model = onnx.helper.make_model(graph)

print('Model: %s'%onnx_model)

fn = 'ABCD.onnx'
onnx.save(onnx_model, fn)

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