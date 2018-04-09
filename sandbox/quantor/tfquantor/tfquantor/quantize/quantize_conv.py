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
_QUANTIZABLE_TYPES = {'Conv2D', 'MatMul', 'DepthwiseConv2dNative'}

# Activations that are supported by the quantization rewrite.
_ACTIVATION_TYPES = {'Relu', 'Relu6', 'Identity'}

# Weight types that are supported by the quantization rewrite.
_WEIGHT_TYPES = {'Variable', 'VariableV2', 'VarHandleOp', 'Const', 'Transpose'}


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

    # Quantize the layer (treat it as activation).
    consumer_ops = input_to_ops_map.ConsumerOperations(
        layer_match.layer_op)
    add_context = context
    quantize._InsertQuantOp(
        add_context,
        'act_quant',
        layer_match.layer_op,
        consumer_ops,
        is_training,
        moving_avg=True,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        vars_collection=vars_collection,
        bits=activation_bits,
        init_min=0.0)


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
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          input_pattern,
          graph_matcher.OneofPattern([weight_pattern, weight_var_pattern, folded_weight_pattern])
      ])

  # Match the convolution op where there is no bias and activation
  layer_matcher = graph_matcher.GraphMatcher(layer_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    yield _LayerMatch(layer_op, weight_tensor)


class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, layer_op, weight_tensor):
    self._layer_op = layer_op
    self._weight_tensor = weight_tensor

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def weight_tensor(self):
    return self._weight_tensor
