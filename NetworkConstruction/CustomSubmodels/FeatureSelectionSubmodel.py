import numpy as np
import tensorflow as tf
from ..CustomSubmodel import CustomSubmodel
from ..CustomLayers.DenseLayer import DenseLayer
from ..CustomLayers.OneToOneLayer import OneToOneLayer

class FeatureSelectionSubmodel(CustomSubmodel):
    def __init__(self, model, **kwargs):
        super(FeatureSelectionSubmodel, self).__init__(model, **kwargs)
        self.build_selec = False
        self.selec_weights = None
        self.output_dim = kwargs['outdim']
        if kwargs['build_selec']:
            self.build_selec = True
            self.selec_layer = OneToOneLayer(None, model = model)

        self.std_hidden_s = kwargs['hidden_sizes']

        self.std_layers = [DenseLayer(el, model=model, func=tf.nn.relu) for el in self.std_hidden_s]
        self.out_layer = DenseLayer(self.output_dim, model = model, func=tf.identity)

    def forward(self, x, dropout):
        out = x
        outs = []
        if self.build_selec:
            out = self.selec_layer(x)
        outs.append(out)
        for stdl in self.std_layers:
            out = stdl(out)
            outs.append(out)
            out = tf.nn.dropout(out, rate = dropout)

        out = self.out_layer(out)

        return out, outs

    def get_selec_weights(self):
        if self.build_selec:
            return self.selec_layer.selec_weights
        else:
            return 1.0

    def get_deriv_importances(self, x, output_imp_prior):
        with tf.GradientTape() as tape:
            tape.watch(x)
            out, _ = self.forward(x, 0.0)
            out = out * output_imp_prior
        return tape.gradient(out, x)

    def imp_pass(self, list_of_inputs):
        next_layer_imp = list_of_inputs[0]  # shape = [batch, previous_size[0]]
        current_output = list_of_inputs[1]  # shape = [batch, size]
        weights = list_of_inputs[2]  # shape = [size x previous_size[0]]

        weights_exp = tf.tile(tf.expand_dims(weights, axis=0), [tf.shape(next_layer_imp)[0], 1, 1])
        current_output_exp = tf.expand_dims(current_output, axis=2)
        global_m = tf.multiply(current_output_exp, tf.maximum(weights_exp, 0))

        denominators = tf.reduce_sum(global_m, axis=1)
        global_multiplier = tf.divide(next_layer_imp, tf.clip_by_value(denominators,
                                                                       clip_value_min=1e-10,
                                                                       clip_value_max=np.infty))
        final_weighted_matrix = tf.multiply(global_m, tf.expand_dims(global_multiplier, axis=1))
        layer_imp = tf.reduce_sum(final_weighted_matrix, axis=2)
        return layer_imp

    def get_activ_importances(self, x, output_imp_prior):
        out, outs = self.forward(x, 0.0)
        last_layer_imp = self.imp_pass([output_imp_prior, outs[-1], self.out_layer.kernel])
        for relu, out_relu in zip(self.std_layers[::-1], outs[:-1][::-1]):
            last_layer_imp = self.imp_pass([last_layer_imp, out_relu, relu.kernel])
        if self.build_selec:
            last_layer_imp = last_layer_imp * self.selec_layer.selec_weights

        return last_layer_imp

    def get_importances(self, x, output_imp_prior):
        return {'gradient':self.get_deriv_importances(x, output_imp_prior),
                'lrp':self.get_activ_importances(x, output_imp_prior)}
