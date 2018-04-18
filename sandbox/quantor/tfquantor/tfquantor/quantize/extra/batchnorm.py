from tfquantor.quantize import graph_matcher
from tfquantor.quantize.extra.common import ExtraLayerMatch

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
    yield ExtraLayerMatch([], [mul_op, act_op])
