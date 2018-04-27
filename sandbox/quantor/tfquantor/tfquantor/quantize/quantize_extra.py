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
from tfquantor.quantize import graph_matcher
from tfquantor.quantize import input_to_ops
from tfquantor.quantize import quantize
from tfquantor.quantize.extra import *
from tfquantor.quantize import quant_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

_EXTRA_QUANTIZE_OPTION_LIST=['deconvolution', 'convolution', 'batchnorm', 'concat', 'leaky_relu']


def Quantize(graph,
             is_training,
             extra_option,
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
  if extra_option is None:
    return

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
  total_generator = itertools.chain(default_generator(graph))
  if 'batchnorm' in option_list or 'all' in option_list:
    total_generator = itertools.chain(total_generator, batchnorm_generator(graph))
  if 'deconvolution' in option_list or 'all' in option_list:
    total_generator = itertools.chain(total_generator, deconvolution_generator(graph))
  if 'convolution' in option_list or 'all' in option_list:
    total_generator = itertools.chain(total_generator, convolution_generator(graph))
  if 'concat' in option_list or 'all' in option_list:
    total_generator = itertools.chain(total_generator, concat_generator(graph))
  if 'leaky_relu' in option_list or 'all' in option_list:
    total_generator = itertools.chain(total_generator, leaky_relu_generator(graph))
  return total_generator
