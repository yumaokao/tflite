from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from tensorflow.contrib.quantize.python import quantize_graph
from .quantize import quantize_graph


def create_training_graph(*args, **kwargs):
  return quantize_graph.create_training_graph(*args, **kwargs)


def create_eval_graph(*args, **kwargs):
  return quantize_graph.create_eval_graph(*args, **kwargs)
