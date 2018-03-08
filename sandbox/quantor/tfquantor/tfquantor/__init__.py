from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.quantize.python import quantize_graph
from .quantize import quantize_graph, fold_batch_norms, quantize
from .quantize import copy_graph


def create_training_graph(*args, **kwargs):
  quantize_graph.create_training_graph(*args, **kwargs)

def create_eval_graph(*args, **kwargs):
  quantize_graph.create_eval_graph(*args, **kwargs)

def create_training_graph_and_return(input_graph=None, quant_delay=0):

  if quant_delay == 0:
    freeze_bn_delay = int(2e5)
  else:
    freeze_bn_delay = quant_delay + int(2e6)
  weight_bits=8
  activation_bits=8

  g = copy_graph.CopyGraph(input_graph)
  with g.as_default():
    fold_batch_norms.FoldBatchNorms(
        g,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=False)
    quantize.Quantize(
        g,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
  return g

def create_eval_graph_and_return(input_graph, *args, **kwargs):
  g = copy_graph.CopyGraph(input_graph)
  quantize_graph.create_eval_graph(g, *args, **kwargs)
  return g
