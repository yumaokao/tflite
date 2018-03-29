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

# Activations that are supported by the quantization rewrite.
_ACTIVATION_TYPES = {'Relu', 'Relu6', 'Identity'}


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
    # Quantize the mul.
    context = quantize._GetContextFromOp(layer_match.layer_op)
    quantize._InsertQuantOp(
        context,
        'mul_quant',
        layer_match.mul_op,
        [layer_match.add_op],
        is_training,
        moving_avg=True,
        ema_decay=ema_decay,
        quant_delay=quant_delay,
        vars_collection=vars_collection,
        bits=activation_bits)

    # Quantize the activations.
    quantize._InsertQuantOp(
        context,
        'act_quant',
        layer_match.activation_op,
        input_to_ops_map.ConsumerOperations(layer_match.activation_op),
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
    yield _LayerMatch(layer_op, mul_op, add_op, act_op)


class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, layer_op, mul_op, add_op, activation_op):
    self._layer_op = layer_op
    self._mul_op = mul_op
    self._add_op = add_op
    self._activation_op = activation_op

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def mul_op(self):
    return self._mul_op

  @property
  def add_op(self):
    return self._add_op

  @property
  def activation_op(self):
    return self._activation_op
