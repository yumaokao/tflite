class ExtraLayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, last_value_quant_ops, moving_avg_quant_ops, last_value_quant_names=None, moving_avg_quant_names=None):
    self._last_value_quant_ops = last_value_quant_ops # ops that requires a following last value quant
    self._moving_avg_quant_ops = moving_avg_quant_ops # ops that requires a following moving avg quant
    if last_value_quant_names is None:
      self._last_value_quant_names = [None] * len(last_value_quant_ops)
    else:
      self._last_value_quant_names = last_value_quant_names

    if moving_avg_quant_names is None:
      self._moving_avg_quant_names = [None] * len(moving_avg_quant_ops)
    else:
      self._moving_avg_quant_names = moving_avg_quant_names

    # Checks
    assert len(self._last_value_quant_names) == len(self._last_value_quant_ops)
    assert len(self._moving_avg_quant_names) == len(self._moving_avg_quant_ops)

  @property
  def last_value_quant_ops(self):
    return self._last_value_quant_ops

  @property
  def moving_avg_quant_ops(self):
    return self._moving_avg_quant_ops

  @property
  def last_value_quant_names(self):
    return self._last_value_quant_names

  @property
  def moving_avg_quant_names(self):
    return self._moving_avg_quant_names
