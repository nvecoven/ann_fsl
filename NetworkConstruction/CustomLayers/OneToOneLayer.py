import tensorflow as tf
from .. import CustomLayer

class OneToOneLayer(CustomLayer.CustomLayer):
    def __init__(self, output_dim, model, var_list = None, **kwargs):
        self.output_dim = output_dim
        super(OneToOneLayer, self).__init__(output_dim, model, var_list, **kwargs)

    def build(self, input_shape):
        self.selec_weights = self.add_weight(name="sweihts", shape=(input_shape[1],),
                                             initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                             dtype = tf.float32)
        super(OneToOneLayer, self).build(input_shape)

    def call(self, x):
        layer_out = x * self.selec_weights
        return layer_out
