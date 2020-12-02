import onnx
import numpy as np


A = onnx.helper.make_node(
    'Sin',
    inputs=['X'],
    outputs=['Y']
)

print('Node: %s'%A)

X = onnx.helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1])
Y = onnx.helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1])

graph = onnx.helper.make_graph(
        nodes=[A],
        name='SinNode',
        inputs=[X],
        outputs=[Y])
#kwargs[str('producer_name')] = 'pgtest'
onnx_model = onnx.helper.make_model(graph)

print('Model: %s'%onnx_model)


fn = 'Sin.onnx'
onnx.save(onnx_model, fn)

print('Done! Exported to: %s'%fn)


def info(a):
    print('Info: %s (%s), %s'%(a.name, a.type, a.shape))

import onnxruntime as rt

sess = rt.InferenceSession(fn)
info(sess.get_inputs()[0])
info(sess.get_outputs()[0])


import numpy
x = numpy.array([1.0],np.float32)
print('Inputting: %s'%x)

res = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
print('Output calculated by onnxruntime:  %s'%res)


print('Done! ONNX inference')
    
