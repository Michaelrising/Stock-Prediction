from MegaModel import Mega, MegaEncoderLayer
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from cv import *
from utils import *


class MegaPredictor(keras.Model):
    def __init__(self,
                features, # original name is dim 512
                mid_feature, # 64
                hidden_dim,
                out_dim=1,
                chunk_size=-1,
                ff_mult = 2
                ):
        super(MegaPredictor, self).__init__()
        self.encoder = keras.Sequential()
        # self.encoder.add(keras.layers.Input(shape=(features, )))
        self.encoder.add(keras.layers.GaussianNoise(0.01))
        self.encoder.add(keras.layers.Dense(mid_feature, activation='swish'))
        self.encoder.add(MegaEncoderLayer(features=mid_feature, chunk_size=chunk_size, ff_mult=ff_mult))

        # self.decoder = keras.Sequential()
        # self.decoder.add(keras.layers.Dense(mid_feature, activation='swish'))
        # self.decoder.add(keras.layers.Dropout(0.25))
        # self.decoder.add(keras.layers.Dense(features, activation='swish'))

        self.concat = keras.layers.Concatenate()
        self.norm = keras.layers.LayerNormalization()

        self.out = keras.Sequential([keras.layers.Dense(hidden_dim, activation=keras.activations.gelu),
                                     keras.layers.LayerNormalization(),
                                    keras.layers.Dense(out_dim)
                                    ])

    @tf.function
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.concat([x, inputs])
        x = self.norm(x)
        y = tf.squeeze(self.out(x))
        return y

    def compute_loss(self,x, y_true, y_pred, sample_weight=None):
        loss = self.compiled_loss(y_true, y_pred, sample_weight=sample_weight)
        # loss += tf.reduce_mean(tf.constant(0.1)/tf.math.maximum(IC(y_true, y_pred, sample_weight), 0.0001))
        return loss

    def train_step(self, data):
        x, (y, mask) = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight=mask)
        # self.add_metric(IC(y, y_pred, mask), name='IC', aggregation='mean')
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)

        # Update metrics (includes the metric that tracks the loss)
        _ = self.compute_metrics(x, y, y_pred, sample_weight=mask)
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        x, (y, mask) = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(x, y, y_pred, sample_weight=mask)
        # self.add_metric(IC(y, y_pred, mask), name='IC', aggregation='mean')
        # Update metrics (includes the metric that tracks the loss)
        _ = self.compute_metrics(x, y, y_pred, sample_weight=mask)
        return {m.name: m.result() for m in self.metrics}

