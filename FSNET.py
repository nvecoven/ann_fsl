from NetworkConstruction.CustomModel import CustomModel
from NetworkConstruction.CustomSubmodels.FeatureSelectionSubmodel import FeatureSelectionSubmodel
import tensorflow as tf
import numpy as np

class FSNET(CustomModel):
    def define_layers(self, params):
        self.output_dim = params['outdim']

        self.net = FeatureSelectionSubmodel(self, hidden_sizes=params['hidden'],
                                            build_selec = params['build_selec'],
                                            outdim = self.output_dim)
        to_checkpoint = {'net':self.net}

        return {**to_checkpoint}

    def create_optimizer(self, params):
        self.opt = tf.optimizers.Adam(learning_rate=params['lr'])


    def cross_entropy(self, x, y, dropout):
        out_net, _ = self.net.forward(x, dropout)
        return tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_net), [-1, 1])

    def mse(self, x, y, dropout):
        out_net, _ = self.net.forward(x, dropout)
        return tf.reshape(tf.reduce_mean((out_net-y)**2, axis = 1), [-1, 1])

    def accuracy(self, x, y, dropout):
        pred, _ = self.net.forward(x, dropout)
        argmax = tf.argmax(pred, axis=1)
        true = tf.argmax(y, axis=1)
        x = tf.cast(tf.equal(argmax, true), tf.float32)
        return tf.reshape(x, [-1, 1])


    def l1(self, x, alpha, alpha2):
        return tf.ones([tf.shape(x)[0],1]) * alpha * tf.reduce_mean(tf.abs(self.net.get_selec_weights()))

    def l2(self, x, alpha, alpha2):
        return tf.ones([tf.shape(x)[0],1]) * alpha2 * tf.reduce_mean(tf.square(self.net.get_selec_weights()))

    def var(self, x, alpha, alpha2):
        x = self.net.selec_layer(x)
        sq = tf.reduce_mean(tf.square(x), axis=0)
        means = tf.square(tf.reduce_mean(x, axis=0))
        var = tf.subtract(sq, means)
        mean_var = tf.reduce_mean(var)
        return tf.ones([tf.shape(x)[0],1]) * mean_var * alpha2

    def l1l2(self, x, alpha, alpha2):
        return self.l1(x, alpha, alpha2) + self.l2(x, alpha, alpha2)

    def noregul(self, x, alpha, alpha2):
        return tf.zeros([tf.shape(x)[0], 1])

    def l1var(self, x, alpha, alpha2):
        return self.l1(x, alpha, alpha2) + self.var(x, alpha, alpha2)

    def global_loss(self, x, y, dropout, score_loss, regul_loss):
        return {'performance_loss':score_loss(x, y, dropout),
                'regularization_loss':regul_loss(x)}

    #@tf.function
    def train_loss(self, samples, state, dropout, score_loss, regul_loss):
        return self.global_loss(samples['inputs'], samples['outputs'], dropout,
                                score_loss, regul_loss), None

    def get_prediction(self, x):
        return self.net.forward(x, 0.0)

    @tf.function
    def get_importances(self, samples, state, out_imp_prior):
        out_imp_prior = tf.expand_dims(out_imp_prior, axis = 0)
        out_imp_prior = tf.tile(out_imp_prior, [tf.shape(samples['inputs'])[0], 1])
        d = self.net.get_importances(samples['inputs'], out_imp_prior)
        d['selec'] = tf.ones_like(samples['inputs']) * self.net.get_selec_weights()
        return d, None