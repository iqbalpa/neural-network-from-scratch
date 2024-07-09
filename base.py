class Layer:
  def __init__(self):
    self.input = None
    self.output = None

  def forward_prop(self, input_data):
    raise NotImplementedError

  def backward_prop(self, output_error, learning_rate):
    raise NotImplementedError