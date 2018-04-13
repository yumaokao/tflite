from tfquantor.quantize import graph_matcher
from tfquantor.quantize.extra.common import ExtraLayerMatch


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
    yield ExtraLayerMatch([weight_op], [bypass_op, bias_add_op])

  bypass_matcher_b = graph_matcher.GraphMatcher(bypass_pattern_b)
  for match_result in bypass_matcher_b.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern_b)
    yield ExtraLayerMatch([weight_op], [bypass_op, bias_add_op])

  # Bypass add without pool fc and deconv
  bypass_matcher_c = graph_matcher.GraphMatcher(bypass_pattern_c)
  for match_result in bypass_matcher_c.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bypass_op = match_result.get_op(bypass_pattern_d)
    yield ExtraLayerMatch([weight_op], [bypass_op, layer_op])

  bypass_matcher_d = graph_matcher.GraphMatcher(bypass_pattern_d)
  for match_result in bypass_matcher_d.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bypass_op = match_result.get_op(bypass_pattern_d)
    yield ExtraLayerMatch([weight_op], [bypass_op, layer_op])

  # BiasAdd without a following activation op
  bias_add_matcher = graph_matcher.GraphMatcher(bias_add_pattern)
  for match_result in bias_add_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    yield ExtraLayerMatch([weight_op], [bias_add_op])

  # Without a BiasAdd op
  layer_matcher = graph_matcher.GraphMatcher(layer_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_op = match_result.get_op(weight_pattern)
    if weight_op is None:
      weight_op = match_result.get_op(weight_var_pattern)
    yield ExtraLayerMatch([weight_op], [layer_op])


