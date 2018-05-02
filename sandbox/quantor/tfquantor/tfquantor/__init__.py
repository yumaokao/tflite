from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.quantize.python import quantize_graph
from tensorflow.python.framework import ops
from .quantize import quantize_graph, fold_batch_norms, quantize, quantize_extra
from .quantize import input_to_ops
from .quantize import copy_graph
from .version  import __version__


# APIs for post-quantize
def create_direct_quant_training_graph(input_graph=None,
                                       quant_delay=0,
                                       inplace=True,
                                       extra_quantize_option=None,
                                       fold_batchnorms=True):
  weight_bits = 8
  activation_bits = 8
  freeze_bn_delay = None

  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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
                                   extra_quantize_option=None,
                                   fold_batchnorms=True):

  weight_bits = 8
  activation_bits = 8

  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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
                                                    extra_quantize_option=None,
                                                    fold_batchnorms=True):
  freeze_bn_delay = None

  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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
                                                extra_quantize_option=None,
                                                fold_batchnorms=True):
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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
                          extra_quantize_option=None,
                          fold_batchnorms=True):
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
    if fold_batchnorms:
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
                      extra_quantize_option=None,
                      fold_batchnorms=True):
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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
                                       extra_quantize_option=None,
                                       fold_batchnorms=True):
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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
                                   extra_quantize_option=None,
                                   fold_batchnorms=True):
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():
    if fold_batchnorms:
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


# Other APIs
def create_custom_eval_graph(target_nodes,
                             input_graph=None,
                             inplace=True):
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  for node in target_nodes:
    # For each node, create a following moving average fakequant node
    add_custom_fakequant_node(node,
                              input_graph=input_graph,
                              is_training=False,
                              inplace=True)

  if inplace is False:
    return input_graph


def create_custom_training_graph(target_nodes,
                                 input_graph=None,
                                 quant_delay=0,
                                 inplace=True):
  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  for node in target_nodes:
    # For each node, create a following moving average fakequant node
    add_custom_fakequant_node(node,
                              input_graph=input_graph,
                              quant_delay=quant_delay,
                              is_training=True,
                              inplace=True)

  if inplace is False:
    return input_graph


def add_custom_fakequant_node(target_node,
                              input_graph=None,
                              context=None,
                              quantize_bits=8,
                              moving_avg=True,
                              ema_decay=0.999,
                              quant_delay=None,
                              is_training=False,
                              inplace=True):

  if input_graph is None:
    input_graph = ops.get_default_graph()

  if inplace is False:
    input_graph = copy_graph.CopyGraph(input_graph)

  with input_graph.as_default():

    input_to_ops_map = input_to_ops.InputToOps(input_graph)
    op = input_graph.get_operation_by_name(target_node)

    if context is None:
      context = quantize._GetContextFromOp(op)

    consumer_ops = input_to_ops_map.ConsumerOperations(op)
    quantize._InsertQuantOp(
        context,
        'custom_quant',
        op,
        consumer_ops,
        is_training=is_training,
        moving_avg=moving_avg,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        vars_collection=ops.GraphKeys.MOVING_AVERAGE_VARIABLES,
        bits=quantize_bits,
        init_min=0.0)

  if inplace is False:
    return input_graph
