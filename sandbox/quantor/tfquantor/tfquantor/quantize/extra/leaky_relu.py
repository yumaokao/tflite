from tfquantor.quantize import graph_matcher
from tfquantor.quantize.extra.common import ExtraLayerMatch

def leaky_relu_generator(graph):

  # leaky_relu = tf.maximum(LEAKY_ALPHA * x, x)
  input_1_pattern = graph_matcher.OpTypePattern('*')
  input_2_pattern = graph_matcher.OpTypePattern('*')
  const_pattern = graph_matcher.OpTypePattern('Const')
  mul_1_pattern = graph_matcher.OpTypePattern(
      'Mul', inputs=[input_1_pattern, const_pattern])
  mul_2_pattern = graph_matcher.OpTypePattern(
      'Mul', inputs=[const_pattern, input_1_pattern])
  max_1_pattern = graph_matcher.OpTypePattern(
      'Maximum', inputs=[
        graph_matcher.OneofPattern([mul_1_pattern, mul_2_pattern]),
        input_2_pattern
        ])
  max_2_pattern = graph_matcher.OpTypePattern(
      'Maximum', inputs=[
        input_2_pattern,
        graph_matcher.OneofPattern([mul_1_pattern, mul_2_pattern])
        ])

  max_1_matcher = graph_matcher.GraphMatcher(max_1_pattern)
  for match_result in max_1_matcher.match_graph(graph):
    input_1_op = match_result.get_op(input_1_pattern)
    input_2_op = match_result.get_op(input_2_pattern)
    max_op = match_result.get_op(max_1_pattern)
    if input_1_op.name == input_2_op.name:
      print('FIND LEAKY_RELU TYPE 1 OP: {}'.format(max_op.name))
      yield ExtraLayerMatch([], [max_op])

  max_2_matcher = graph_matcher.GraphMatcher(max_2_pattern)
  for match_result in max_2_matcher.match_graph(graph):
    input_1_op = match_result.get_op(input_1_pattern)
    input_2_op = match_result.get_op(input_2_pattern)
    max_op = match_result.get_op(max_2_pattern)
    if input_1_op.name == input_2_op.name:
      print('FIND LEAKY_RELU TYPE 2 OP: {}'.format(max_op.name))
      yield ExtraLayerMatch([], [max_op])
