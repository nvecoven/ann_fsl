import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, model, var_list = None, **kwargs):
        self.output_dim = output_dim
        self.var_list = var_list
        self.model = model
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not self.var_list is None:
            for vl in self.var_list:
                if not (vl in self.model.variables_d):
                    self.model.variables_d[vl] = self.variables
                else:
                    self.model.variables_d[vl] += self.variables
        self.model.variables += self.variables

        super(CustomLayer, self).build(input_shape)

    def call(self, x):
        raise NotImplementedError("Please Implement this method")

    def zero_state(self, batch_size):
        raise NotImplementedError("Please Implement this method")

