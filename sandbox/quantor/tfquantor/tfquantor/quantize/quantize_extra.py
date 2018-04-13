# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Logic to update a TensorFlow model graph with quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import itertools
from tensorflow.contrib import graph_editor
from tfquantor.quantize import common
from tensorflow.contrib.quantize.python import graph_matcher
from tfquantor.quantize import input_to_ops
from tfquantor.quantize import quantize
from tfquantor.quantize import quant_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

_EXTRA_QUANTIZE_OPTION_LIST=['deconvolution', 'convolution', 'batchnorm']

def _all_extra_option():
  return ' '.join(_EXTRA_QUANTIZE_OPTION_LIST)

def Quantize(graph,
             is_training,
             extra_option=_all_extra_option(),
             weight_bits=8,
             activation_bits=8,
             ema_decay=0.999,
             quant_delay=None,
             vars_collection=ops.GraphKeys.MOVING_AVERAGE_VARIABLES):
  """Updates graph with quantization operations.

  Args:
    graph: Graph to modify.
    is_training: Whether quantizing training graph or eval graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
  Raises:
    ValueError: When quantization fails.
  """
  input_to_ops_map = input_to_ops.InputToOps(graph)
  for layer_match in _FindLayersToQuantize(graph, extra_option):

    # Quantize the weights.
    for weight_op in layer_match.weight_ops:
      context = quantize._GetContextFromOp(weight_op)
      consumer_ops = input_to_ops_map.ConsumerOperations(weight_op)
      quantize._InsertQuantOp(
          context,
          'weights_quant',
          weight_op,
          consumer_ops,
          is_training,
          moving_avg=False,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          narrow_range=True,
          vars_collection=vars_collection,
          bits=weight_bits)

    # Quantize the activations.
    for act_op in layer_match.act_ops:
      context = quantize._GetContextFromOp(act_op)
      consumer_ops = input_to_ops_map.ConsumerOperations(act_op)
      quantize._InsertQuantOp(
          context,
          'act_quant',
          act_op,
          consumer_ops,
          is_training,
          moving_avg=True,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          vars_collection=vars_collection,
          bits=activation_bits,
          init_min=0.0)

def _FindLayersToQuantize(graph, extra_option):
  option_list = extra_option.split()
  total_generator = itertools.chain(empty_generator(graph))
  if 'batchnorm' in option_list:
    total_generator = itertools.chain(total_generator, batchnorm_generator(graph))
  if 'deconvolution' in option_list:
    total_generator = itertools.chain(total_generator, deconvolution_generator(graph))
  if 'convolution' in option_list:
    total_generator = itertools.chain(total_generator, convolution_generator(graph))
  return total_generator

def empty_generator(graph):
  return
  yield

def batchnorm_generator(graph):
  _ACTIVATION_TYPES = {'Relu', 'Relu6', 'Identity'}

  input_pattern = graph_matcher.OpTypePattern('*')

  # Match the preact pattern in resnet_v2 network
  batchnorm_const_pattern = graph_matcher.OpTypePattern('Const')
  batchnorm_mul_pattern = graph_matcher.OpTypePattern(
      'Mul', inputs=[input_pattern, batchnorm_const_pattern])
  batchnorm_add_pattern = graph_matcher.OpTypePattern(
      'Add', inputs=[batchnorm_mul_pattern, batchnorm_const_pattern])
  preact_pattern = graph_matcher.OpTypePattern(
      '|'.join(_ACTIVATION_TYPES), inputs=[batchnorm_add_pattern])

  preact_matcher = graph_matcher.GraphMatcher(preact_pattern)
  for match_result in preact_matcher.match_graph(graph):
    layer_op = match_result.get_op(batchnorm_mul_pattern)
    mul_op = match_result.get_op(batchnorm_mul_pattern)
    add_op = match_result.get_op(batchnorm_add_pattern)
    act_op = match_result.get_op(preact_pattern)
    # No need to quantize add_op since it will be folded by the following act_op
    yield _LayerMatch([], [mul_op, act_op])

def convolution_generator(graph):
  _CONV_TYPES = {'Conv2D', 'DepthwiseConv2dNative'}
  _ACTIVATION_TYPES = {'Relu', 'Relu6', 'Identity'}
  _WEIGHT_TYPES = {'Variable', 'VariableV2', 'VarHandleOp', 'Const', 'Transpose'}

  input_pattern = graph_matcher.OpTypePattern('*')

  weight_var_pattern = graph_matcher.OpTypePattern('|'.join(_WEIGHT_TYPES))
  weight_pattern = graph_matcher.OpTypePattern(
      'Identity|ReadVariableOp', inputs=[weight_var_pattern])
  folded_weight_pattern = graph_matcher.OpTypePattern('Mul')

  # The weights inputs to the layer operation can either be from the Variable or
  # the folded weight (Mul).
  layer_pattern = graph_matcher.OpTypePattern(
      '|'.join(_CONV_TYPES),
      inputs=[
          input_pattern,
          graph_matcher.OneofPattern([weight_pattern, weight_var_pattern, folded_weight_pattern])
      ])

  # Match the convolution op where there is no bias and activation
  # Since convolution ops with bias or activation will all be matched
  # by the previous graph matcher passes
  layer_matcher = graph_matcher.GraphMatcher(layer_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(folded_weight_pattern)
    yield _LayerMatch([weight_op], [layer_op])

def deconvolution_generator(graph):
  _DECONV_TYPES = {'Conv2DBackpropInput'}
  _WEIGHT_TYPES = {'Variable', 'VariableV2', 'VarHandleOp', 'Const', 'Transpose'}

  input_pattern = graph_matcher.OpTypePattern('*')
  weight_var_pattern = graph_matcher.OpTypePattern('|'.join(_WEIGHT_TYPES))
  weight_pattern = graph_matcher.OpTypePattern(
      'Identity|ReadVariableOp', inputs=[weight_var_pattern])

  # The weights inputs to the layer operation can either be from the Variable or
  # the folded weight (Mul).
  layer_pattern = graph_matcher.OpTypePattern(
      '|'.join(_DECONV_TYPES),
      inputs=[
          '*',
          graph_matcher.OneofPattern([weight_pattern, weight_var_pattern]),
          input_pattern
      ])

  bias_add_pattern = graph_matcher.OpTypePattern(
      'Add|BiasAdd', inputs=[layer_pattern, '*'])

  # The bias can come from the bias add or the folded bias add.
  bypass_pattern_a = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          bias_add_pattern, '*'
      ])
  bypass_pattern_b = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          '*',
          bias_add_pattern
      ])

  # The bypass add with fake quantized conv and deconv for FCN
  pool_fc_biasadd_pattern = graph_matcher.OpTypePattern('BiasAdd')
  pool_fc_pattern = graph_matcher.OpTypePattern(
      'FakeQuantWithMinMaxVars', inputs=[pool_fc_biasadd_pattern, '*', '*'])
  bypass_pattern_c = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          layer_pattern,
          pool_fc_pattern
      ])
  bypass_pattern_d = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          pool_fc_pattern,
          layer_pattern
      ])

  # Bypass add without a following activation op
  bypass_matcher_a = graph_matcher.GraphMatcher(bypass_pattern_a)
  for match_result in bypass_matcher_a.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern_a)
    yield _LayerMatch([weight_op], [bypass_op, bias_add_op])

  bypass_matcher_b = graph_matcher.GraphMatcher(bypass_pattern_b)
  for match_result in bypass_matcher_b.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern_b)
    yield _LayerMatch([weight_op], [bypass_op, bias_add_op])

  # Bypass add without pool fc and deconv
  bypass_matcher_c = graph_matcher.GraphMatcher(bypass_pattern_c)
  for match_result in bypass_matcher_c.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bypass_op = match_result.get_op(bypass_pattern_d)
    yield _LayerMatch([weight_op], [bypass_op, layer_op])

  bypass_matcher_d = graph_matcher.GraphMatcher(bypass_pattern_d)
  for match_result in bypass_matcher_d.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bypass_op = match_result.get_op(bypass_pattern_d)
    yield _LayerMatch([weight_op], [bypass_op, layer_op])

  # BiasAdd without a following activation op
  bias_add_matcher = graph_matcher.GraphMatcher(bias_add_pattern)
  for match_result in bias_add_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    yield _LayerMatch([weight_op], [bias_add_op])

  # Without a BiasAdd op
  layer_matcher = graph_matcher.GraphMatcher(layer_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    yield _LayerMatch([weight_op], [layer_op])


class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, weight_ops, act_ops):
    self._weight_ops = weight_ops # ops that requires a following weight quant
    self._act_ops = act_ops # ops that requires a following act quant

  @property
  def weight_ops(self):
    return self._weight_ops

  @property
  def act_ops(self):
    return self._act_ops
