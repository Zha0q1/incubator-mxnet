
import os
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet.contrib import onnx as onnx_mxnet
import onnxruntime as rt


mx.random.seed(112233)

batch = 5
seq_length = 16
inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32')
token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32')
valid_length = mx.nd.array([seq_length] * batch, dtype='float32')

print(inputs)

onnx_file = 'bert_model/mx_bert_layer12.onnx'

in_tensors = [inputs, token_types, valid_length]
sess = rt.InferenceSession(onnx_file)
input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
print(input_dict)
pred = sess.run(None, input_dict)
print(pred[0], pred[0].shape)

print(pred[1], pred[1].shape)

