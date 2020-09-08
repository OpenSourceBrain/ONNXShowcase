'''
Based on https://github.com/onnx/keras-onnx/blob/master/tutorial/TensorFlow_Keras_EfficientNet.ipynb
'''

import numpy as np
import efficientnet.tfkeras as efn
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from efficientnet.preprocessing import center_crop_and_resize
from skimage.io import imread
model = efn.EfficientNetB0(weights='imagenet')

print('Loaded...')

import matplotlib.pyplot as plt
image = imread('panda.jpg')
image_size = model.input_shape[1]
plt.imshow(image, interpolation='nearest')
plt.show()
x = center_crop_and_resize(image, image_size=image_size)

x = preprocess_input(x, mode='torch')
inputs = np.expand_dims(x, 0)
expected = model.predict(inputs)
print(decode_predictions(expected))

import keras2onnx
output_model_path = "keras_efficientNet.onnx"
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, output_model_path)