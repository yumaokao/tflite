from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import ops
from .quantize import quantize_graph, fold_batch_norms, quantize, quantize_extra
from .quantize import copy_graph


# APIs for post-quantize
def create_direct_quant_training_graph(*args, **kwargs):
  weight_bits = 8
  activation_bits = 8
  quant_delay = kwargs['quant_delay'] if 'quant_delay' in kwargs else 0
  freeze_bn_delay = None # This value will not be used
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=False) # Since this is post-quantize
    quantize.Quantize(
        input_graph,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=True,
          extra_option=extra_quantize_option,
          quant_delay=quant_delay,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph


def create_direct_quant_eval_graph(*args, **kwargs):
  weight_bits = 8
  activation_bits = 8
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=None,
        is_training=False)
    quantize.Quantize(
        input_graph,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=False,
          extra_option=extra_quantize_option,
          quant_delay=None,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph

def experimental_create_direct_quant_training_graph(*args, **kwargs):
  weight_bits = kwargs['weight_bits'] if 'weight_bits' in kwargs else 8
  activation_bits = kwargs['activation_bits'] if 'activation_bits' in kwargs else 8
  quant_delay = kwargs['quant_delay'] if 'quant_delay' in kwargs else 0
  freeze_bn_delay = None # This value will not be used
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=False) # Since this is post-quantize
    quantize.Quantize(
        input_graph,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=True,
          extra_option=extra_quantize_option,
          quant_delay=quant_delay,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph


def experimental_create_direct_quant_eval_graph(*args, **kwargs):
  weight_bits = kwargs['weight_bits'] if 'weight_bits' in kwargs else 8
  activation_bits = kwargs['activation_bits'] if 'activation_bits' in kwargs else 8
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=None,
        is_training=False)
    quantize.Quantize(
        input_graph,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=False,
          extra_option=extra_quantize_option,
          quant_delay=None,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph


# APIs for pre-quantize
def create_training_graph(*args, **kwargs):
  weight_bits = 8
  activation_bits = 8
  quant_delay = kwargs['quant_delay'] if 'quant_delay' in kwargs else 0
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if quant_delay == 0:
    freeze_bn_delay = int(2e5)
  else:
    freeze_bn_delay = quant_delay + int(2e6)

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=True)
    quantize.Quantize(
        input_graph,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=True,
          extra_option=extra_quantize_option,
          quant_delay=quant_delay,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph


def create_eval_graph(*args, **kwargs):
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=None,
        is_training=False)
    quantize.Quantize(
        input_graph,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=False,
          extra_option=extra_quantize_option,
          quant_delay=None,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph


def experimental_create_training_graph(*args, **kwargs):
  weight_bits = kwargs['weight_bits'] if 'weight_bits' in kwargs else 8
  activation_bits = kwargs['activation_bits'] if 'activation_bits' in kwargs else 8
  quant_delay = kwargs['quant_delay'] if 'quant_delay' in kwargs else 0
  freeze_bn_delay = kwargs['freeze_bn_delay'] if 'freeze_bn_delay' in kwargs else 0
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=freeze_bn_delay,
        is_training=True)
    quantize.Quantize(
        input_graph,
        is_training=True,
        quant_delay=quant_delay,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=True,
          extra_option=extra_quantize_option,
          quant_delay=quant_delay,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph


def experimental_create_eval_graph(*args, **kwargs):
  weight_bits = kwargs['weight_bits'] if 'weight_bits' in kwargs else 8
  activation_bits = kwargs['activation_bits'] if 'activation_bits' in kwargs else 8
  input_graph = kwargs['input_graph'] if 'input_graph' in kwargs else None
  inplace = kwargs['inplace'] if 'inplace' in kwargs else True
  extra_quantize_option = kwargs['extra_quantize_option'] if 'extra_quantize_option' in kwargs else None
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph,
        freeze_batch_norm_delay=None,
        is_training=False)
    quantize.Quantize(
        input_graph,
        is_training=False,
        quant_delay=None,
        weight_bits=weight_bits,
        activation_bits=activation_bits)
    if extra_quantize_option is not None:
      quantize_extra.Quantize(
          input_graph,
          is_training=False,
          extra_option=extra_quantize_option,
          quant_delay=None,
          weight_bits=weight_bits,
          activation_bits=activation_bits)

  if inplace is False:
    return input_graph
