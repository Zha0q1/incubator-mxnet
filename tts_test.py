import os
import argparse
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from mxnet.contrib import onnx as onnx_mxnet
 
parser = argparse.ArgumentParser(description="Tune/evaluate Bert model")
parser.add_argument("--layer", type=int, default=12,
                    help="Number of layers in BERT model (default: 12)")
parser.add_argument("--task", choices=["classification", "regression", "question_answering"],
                    default="classification",
                    help="specify the model type (default: classification)")
args = parser.parse_args()
 
#ctx = mx.gpu(0)
ctx = mx.cpu(0)
use_pooler = False if args.task == "question_answering" else True
model_name='bert_12_768_12'
dataset='book_corpus_wiki_en_uncased'
bert, _ = nlp.model.get_model(
    name=model_name,
    ctx=ctx,
    dataset_name=dataset,
    pretrained=False,
    use_pooler=use_pooler,
    use_decoder=False,
    use_classifier=False)
model = bert
model.initialize(ctx=ctx)
model.hybridize(static_alloc=True)

mx.random.seed(112233)

batch = 5
seq_length = 16
inputs = mx.nd.random.uniform(0, 30522, shape=(batch, seq_length), dtype='float32', ctx=ctx)
token_types = mx.nd.random.uniform(0, 2, shape=(batch, seq_length), dtype='float32', ctx=ctx)
valid_length = mx.nd.array([seq_length] * batch, dtype='float32', ctx=ctx)
#print(inputs, token_types, valid_length)
seq_encoding, cls_encoding = model(inputs, token_types, valid_length)
print(seq_encoding, seq_encoding.shape)
print(cls_encoding, cls_encoding.shape)

model_dir = f'bert_model'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
 
prefix = '%s/mx_bert_layer%s' % (model_dir, args.layer)
model.export(prefix)
 
sym_file = "%s-symbol.json" % prefix
params_file = "%s-0000.params" % prefix
onnx_file = "%s.onnx" % prefix
input_shapes = [(batch, seq_length), (batch, seq_length), (batch,)]
converted_model_path = onnx_mxnet.export_model(sym_file, params_file, input_shapes, np.float32, onnx_file, verbose=True)
print(converted_model_path)
print(onnx_file)




'''
import onnxruntime as rt
in_tensors = [inputs, token_types, valid_length]
sess = rt.InferenceSession(onnx_file)
input_dict = dict((sess.get_inputs()[i].name, in_tensors[i].asnumpy()) for i in range(len(in_tensors)))
pred = sess.run(None, input_dict)[0]
print(pred)
'''
