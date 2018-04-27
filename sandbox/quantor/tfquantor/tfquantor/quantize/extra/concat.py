from tfquantor.quantize import graph_matcher
from tfquantor.quantize.extra.common import ExtraLayerMatch

def concat_generator(graph):
  _CONCAT_TYPES = {'Concat', 'ConcatV2'}

  concat_pattern = graph_matcher.OpTypePattern('|'.join(_CONCAT_TYPES))

  concat_matcher = graph_matcher.GraphMatcher(concat_pattern)
  for match_result in concat_matcher.match_graph(graph):
    concat_op = match_result.get_op(concat_pattern)
    print('FIND CONCAT OP: {}'.format(concat_op.name))
    yield ExtraLayerMatch([], [concat_op], None, ['concat_quant'])
