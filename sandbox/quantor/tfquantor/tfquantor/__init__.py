from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import ops
from .quantize import quantize_graph, fold_batch_norms, quantize, quantize_extra
from .quantize import copy_graph


# APIs for post-quantize
def create_direct_quant_training_graph(input_graph=None,
                                       quant_delay=0,
                                       inplace=True,
                                       extra_quantize_option=None):
  weight_bits = 8
  activation_bits = 8
  freeze_bn_delay = None

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


def create_direct_quant_eval_graph(input_graph=None,
                                   inplace=True,
                                   extra_quantize_option=None):

  weight_bits = 8
  activation_bits = 8

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

def experimental_create_direct_quant_training_graph(input_graph=None,
                                                    weight_bits=8,
                                                    activation_bits=8,
                                                    quant_delay=0,
                                                    inplace=True,
                                                    extra_quantize_option=None):
  freeze_bn_delay = None

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


def experimental_create_direct_quant_eval_graph(input_graph=None,
                                                weight_bits=8,
                                                activation_bits=8,
                                                inplace=True,
                                                extra_quantize_option=None):
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
def create_training_graph(input_graph=None,
                          quant_delay=0,
                          inplace=True,
                          extra_quantize_option=None):
  weight_bits = 8
  activation_bits = 8

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


def create_eval_graph(input_graph=None,
                      inplace=True,
                      extra_quantize_option=None):
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


def experimental_create_training_graph(input_graph=None,
                                       weight_bits=8,
                                       activation_bits=8,
                                       quant_delay=0,
                                       freeze_bn_delay=int(2e5),
                                       inplace=True,
                                       extra_quantize_option=None):
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


def experimental_create_eval_graph(input_graph=None,
                                   weight_bits=8,
                                   activation_bits=8,
                                   inplace=True,
                                   extra_quantize_option=None):
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
