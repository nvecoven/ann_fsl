import tensorflow as tf
from .. import CustomLayer

class DenseLayer(CustomLayer.CustomLayer):
    def __init__(self, output_dim, model, var_list = None, func = tf.identity, **kwargs):
        self.output_dim = output_dim
        self.activate_function = func
        super(DenseLayer, self).__init__(output_dim, model, var_list, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="k1", shape=(input_shape[1], self.output_dim,),
                                      initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1),
                                      dtype = tf.float32)
        self.bias = self.add_weight(name="b1", shape=(self.output_dim,), dtype = tf.float32, initializer='zeros')

        super(DenseLayer, self).build(input_shape)

    def call(self, x):
        layer_out = self.activate_function(tf.matmul(x, self.kernel) + self.bias)
        return layer_out

    def zero_state(self, batch_size):
        return tf.zeros(shape = (batch_size, self.output_dim), dtype = tf.float32)