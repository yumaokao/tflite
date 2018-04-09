from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import ops
from .quantize import quantize_graph, fold_batch_norms, quantize, quantize_deconv, quantize_batchnorm, quantize_convpool
from .quantize import copy_graph


def create_training_graph(*args, **kwargs):
  quantize_graph.create_training_graph(*args, **kwargs)


def create_eval_graph(*args, **kwargs):
  quantize_graph.create_eval_graph(*args, **kwargs)


def experimental_create_training_graph(*args, **kwargs):
  quantize_graph.experimental_create_training_graph(*args, **kwargs)
  # Default values
  #  weight_bits = kwargs['weight_bits'] if 'weight_bits' in kwargs else 8
  #  activation_bits = kwargs['activation_bits'] if 'activation_bits' in kwargs else 8
  #  quant_delay = kwargs['quant_delay'] if 'quant_delay' in kwargs else 0
  #  input_graph = ops.get_default_graph()
  #  with input_graph.as_default():
    #  quantize_batchnorm.Quantize(
        #  input_graph,
        #  is_training=True,
        #  quant_delay=quant_delay,
        #  weight_bits=weight_bits,
        #  activation_bits=activation_bits)
    #  quantize_convpool.Quantize(
        #  input_graph,
        #  is_training=True,
        #  quant_delay=quant_delay,
        #  weight_bits=weight_bits,
        #  activation_bits=activation_bits)


def experimental_create_eval_graph(*args, **kwargs):
  quantize_graph.experimental_create_eval_graph(*args, **kwargs)
  # Default values
  #  weight_bits = kwargs['weight_bits'] if 'weight_bits' in kwargs else 8
  #  activation_bits = kwargs['activation_bits'] if 'activation_bits' in kwargs else 8
  #  input_graph = ops.get_default_graph()
  #  with input_graph.as_default():
    #  quantize_batchnorm.Quantize(
        #  input_graph,
        #  is_training=False,
        #  quant_delay=None,
        #  weight_bits=weight_bits,
        #  activation_bits=activation_bits)
    #  quantize_convpool.Quantize(
        #  input_graph,
        #  is_training=False,
        #  quant_delay=None,
        #  weight_bits=weight_bits,
        #  activation_bits=activation_bits)


def create_training_graph_and_return(input_graph=None, quant_delay=0, is_batch_norm_training=False):

  # Default values
  weight_bits=8
  activation_bits=8

  if quant_delay == 0:
    freeze_bn_delay = int(2e5)
  else:
    freeze_bn_delay = quant_delay + int(2e6)

  if input_graph is None:
    input_graph = ops.get_default_graph()

  # Clone graph and do the quantization
  g = copy_graph.CopyGraph(input_graph)
  with g.as_default():
    fold_batch_norms.FoldBatchNorms(
        g,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=is_batch_norm_training)
    quantize.Quantize(
        g,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    quantize_deconv.Quantize(
        g,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    quantize_batchnorm.Quantize(
        g,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    quantize_convpool.Quantize(
        g,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
  return g


def create_eval_graph_and_return(input_graph=None):

  # Default values
  weight_bits=8
  activation_bits=8

  if input_graph is None:
    input_graph = ops.get_default_graph()

  # Clone graph and do the quantization
  g = copy_graph.CopyGraph(input_graph)
  with g.as_default():
    fold_batch_norms.FoldBatchNorms(
        g,
        freeze_batch_norm_delay=None,
        is_training=False)
    quantize.Quantize(
        g,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    quantize_deconv.Quantize(
        g,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    quantize_batchnorm.Quantize(
        g,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    quantize_convpool.Quantize(
        g,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
  return g
