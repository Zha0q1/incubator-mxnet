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

ctx = mx.cpu(0)

input_shapes = [(1, 38), (1,)]

sym_file = 'model.exported.joint-symbol.json'
params_file = 'model.exported.joint-0000.params'
onnx_file = 'model.onnx'

converted_model_path = onnx_mxnet.export_model(sym_file, params_file, input_shapes,
                                               np.float32, onnx_file, verbose=True)
