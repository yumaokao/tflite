from tfquantor.quantize import graph_matcher
from tfquantor.quantize.extra.common import ExtraLayerMatch


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
    yield ExtraLayerMatch([weight_op], [layer_op])
