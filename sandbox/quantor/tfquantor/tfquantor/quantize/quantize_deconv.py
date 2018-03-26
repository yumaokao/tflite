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
from tensorflow.contrib import graph_editor
from tfquantor.quantize import common
from tensorflow.contrib.quantize.python import graph_matcher
from tfquantor.quantize import input_to_ops
from tfquantor.quantize import quantize
from tfquantor.quantize import quant_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

# Quantizable operation types that are supported by the quantization rewrite.
_DECONV_TYPES = {'Conv2DBackpropInput'}

# Weight types that are supported by the quantization rewrite.
_WEIGHT_TYPES = {'Variable', 'VariableV2', 'VarHandleOp', 'Const'}


def Quantize(graph,
             is_training,
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
  for layer_match in _FindLayersToQuantize(graph):
    # Quantize the weights.
    context = quantize._GetContextFromOp(layer_match.layer_op)
    quantize._InsertQuantOp(
        context,
        'weights_quant',
        layer_match.weight_tensor.op, [layer_match.layer_op],
        is_training,
        moving_avg=False,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        narrow_range=True,
        vars_collection=vars_collection,
        bits=weight_bits)

    # Quantize the activations.
    consumer_ops = input_to_ops_map.ConsumerOperations(
        layer_match.bias_add_op)
    add_context = context
    if layer_match.bypass_op:
      # (Chia-Lin Yu @ Mediatek 20180323)
      # For cases that only a single hierarchy of namespace is used
      context_match = re.search(r'^(.*)/([^/]+)', context)
      if context_match:
        add_context = context_match.group(1)

    quantize._InsertQuantOp(
        add_context,
        'act_quant',
        layer_match.bias_add_op,
        consumer_ops,
        is_training,
        moving_avg=True,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        vars_collection=vars_collection,
        bits=activation_bits,
        init_min=0.0)

    # Quantize the output to the bypass (if it exists). The input to
    # the bypass is the bias add.
    if layer_match.bypass_op is not None:
      quantize._InsertQuantOp(
          context,
          'add_quant',
          layer_match.bypass_op,
          input_to_ops_map.ConsumerOperations(layer_match.bypass_op),
          is_training,
          moving_avg=True,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          vars_collection=vars_collection,
          bits=activation_bits)


def _FindLayersToQuantize(graph):
  """Matches layers in graph to quantize.

  Args:
    graph: Graph to perform match on.

  Yields:
    _LayerMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_var_pattern = graph_matcher.OpTypePattern('|'.join(_WEIGHT_TYPES))
  weight_pattern = graph_matcher.OpTypePattern(
      'Identity|ReadVariableOp', inputs=[weight_var_pattern])

  folded_weight_pattern = graph_matcher.OpTypePattern('Mul')

  # The weights inputs to the layer operation can either be from the Variable or
  # the folded weight (Mul).
  layer_pattern = graph_matcher.OpTypePattern(
      '|'.join(_DECONV_TYPES),
      inputs=[
          '*',
          graph_matcher.OneofPattern([weight_pattern, folded_weight_pattern]),
          input_pattern
      ])

  folded_bias_mul_pattern = graph_matcher.OpTypePattern(
      'Mul', inputs=[graph_matcher.OpTypePattern('*'), layer_pattern])
  post_layer_op_correction_pattern = graph_matcher.OpTypePattern(
      'Add', inputs=[folded_bias_mul_pattern,
                     graph_matcher.OpTypePattern('*')])
  folded_bias_add_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          post_layer_op_correction_pattern,
          graph_matcher.OpTypePattern('*')
      ])

  bias_add_pattern = graph_matcher.OpTypePattern(
      'Add|BiasAdd', inputs=[layer_pattern, '*'])

  # The bias can come from the bias add or the folded bias add.
  bypass_pattern_a = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          graph_matcher.OneofPattern(
              [bias_add_pattern, folded_bias_add_pattern]), '*'
      ])
  bypass_pattern_b = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          '*',
          graph_matcher.OneofPattern(
              [bias_add_pattern, folded_bias_add_pattern])
      ])

  bypass_matcher_a = graph_matcher.GraphMatcher(bypass_pattern_a)
  for match_result in bypass_matcher_a.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern_a)
    yield _LayerMatch(layer_op, weight_tensor, bypass_op, bias_add_op)

  bypass_matcher_b = graph_matcher.GraphMatcher(bypass_pattern_b)
  for match_result in bypass_matcher_b.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern_b)
    yield _LayerMatch(layer_op, weight_tensor, bypass_op, bias_add_op)

  bias_add_matcher = graph_matcher.GraphMatcher(bias_add_pattern)
  for match_result in bias_add_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    yield _LayerMatch(layer_op, weight_tensor, None, bias_add_op)


class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, layer_op, weight_tensor, bypass_op,
               bias_add_op):
    self._layer_op = layer_op
    self._weight_tensor = weight_tensor
    self._bypass_op = bypass_op
    self._bias_add_op = bias_add_op

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def weight_tensor(self):
    return self._weight_tensor

  @property
  def bypass_op(self):
    return self._bypass_op

  @property
  def bias_add_op(self):
    return self._bias_add_op
