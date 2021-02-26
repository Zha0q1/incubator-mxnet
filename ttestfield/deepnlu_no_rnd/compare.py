import os
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet.contrib import onnx as onnx_mxnet
import onnxruntime as rt
import onnx
import time
from onnxsim import simplify
from mxnet.test_utils import assert_almost_equal
from mxnet import gluon


input_shapes = [(1, 38), (1,)]

sym_file = 'model.exported.joint-symbol.json'
params_file = 'model.exported.joint-0000.params'
onnx_file = 'model.onnx'


ctx = mx.cpu()
model = gluon.nn.SymbolBlock.imports(sym_file, ['data0', 'data1'], params_file, ctx=ctx)

data0 = mx.nd.random.uniform(0, 10, input_shapes[0]).astype('int32').astype('float32')
data1 = mx.nd.array([20]).astype('float32')

print(data0)
print(data1)


mx_out = model(data0, data1)

print('!!~~~~~~~~')
#print(mx_out[4])
print('!!~~~~')




sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = rt.InferenceSession(onnx_file, sess_options)

in_tensors = [data0, data1]
input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
pred_on = sess.run(None, input_dict)

#print(pred_on)


for i in range(5):
    print('mx', mx_out[i])
    print('on', pred_on[i])
    print('~~~~')
    print()

assert_almost_equal(mx_out[0], pred_on[0])
