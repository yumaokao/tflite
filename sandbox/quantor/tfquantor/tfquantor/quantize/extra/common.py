class ExtraLayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, weight_ops, act_ops):
    self._weight_ops = weight_ops # ops that requires a following weight quant
    self._act_ops = act_ops # ops that requires a following act quant

  @property
  def weight_ops(self):
    return self._weight_ops

  @property
  def act_ops(self):
    return self._act_ops
