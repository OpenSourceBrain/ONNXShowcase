FROM tensorflow/tensorflow:2.1.0-py3


RUN apt-get update --fix-missing && apt-get install -y graphviz git vim
RUN pip install pydot graphviz

RUN pip install onnxruntime
RUN pip install git+https://github.com/microsoft/onnxconverter-common
RUN pip install git+https://github.com/onnx/keras-onnx

RUN pip install mxnet
RUN pip install torch
RUN pip install pandas
RUN pip install xgboost
RUN pip install efficientnet

COPY ml_info.py /root

RUN echo '\n\nalias cd..="cd .."\nalias h=history\nalias ll="ls -alt"' >> ~/.bashrc

RUN echo Done!
