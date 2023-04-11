from tensorflow import keras
from utils import plot_cv_indices, IC, mse_corr_loss, PearsonCorrelation

def create_ae_mlp(in_dim, out_dim, hidden_units, dropout_rates, lr=1e-4):
    inp = keras.layers.Input(shape=(in_dim,))
    x0 = keras.layers.BatchNormalization()(inp)

    encoder = keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = keras.layers.Dense(hidden_units[0], kernel_initializer='he_normal')(encoder)
    encoder = keras.layers.BatchNormalization()(encoder)
    encoder = keras.layers.Activation('elu')(encoder) # elu

    decoder = keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = keras.layers.Dense(in_dim * 2, activation='elu', kernel_initializer='he_normal')(decoder)
    decoder = keras.layers.BatchNormalization()(decoder)
    decoder = keras.layers.Dense(in_dim, name='decoder', kernel_initializer='he_normal')(decoder)

    x = keras.layers.Concatenate()([x0, encoder])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = keras.layers.Dense(hidden_units[i], kernel_initializer='he_normal')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Dropout(dropout_rates[i + 2])(x)

    pred = keras.layers.Dense(out_dim, activation='gelu', name='pred', kernel_initializer='he_normal')(x)

    model = keras.models.Model(inputs=inp, outputs=[decoder, pred])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, amsgrad=True, epsilon=1e-8, clipnorm=0.1),
                  loss={'decoder': keras.losses.MeanSquaredError(),
                        'pred': mse_corr_loss,
                        },
                  metrics={'decoder': keras.metrics.MeanAbsoluteError(name='MAE'),
                           'pred': PearsonCorrelation(name='IC')
                           },
                  )

    return model

# class MLP(keras.Model):
#     def __init__(self, in_dim, out_dim, hidden_units, dropout_rates, lr=2e-4):
#         super(MLP, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.hidden_units = hidden_units
#         self.dropout_rates = dropout_rates
#         self.lr = lr
#
#         self.encoder = keras.Sequential()
#         self.encoder.add(keras.layers.Input(shape=(in_dim, )))
#         self.encoder.add(keras.layers.GaussianNoise(dropout_rates[0]))
#         self.encoder.add(keras.layers.Dense(hidden_units[0]))
#         self.encoder.add(keras.layers.BatchNormalization())
#         self.encoder.add(keras.layers.Activation('swish'))
#
#         self.decoder = keras.Sequential()
#         self.decoder.add(keras.layers.Dropout(dropout_rates[1]))
#         self.decoder.add(keras.layers.Dense(in_dim * 2, activation='swish'))
#         self.decoder.add(keras.layers.Dense(in_dim, name='decoder'))
#
#         self.out = keras.Sequential()
#         self.out.add(keras.layers.Dropout(dropout_rates[3]))
#         for i in range(2, len(hidden_units)):
#             self.out.add(keras.layers.Dense(hidden_units[i]))
#             self.out.add(keras.layers.BatchNormalization())
#             self.out.add(keras.layers.Activation('swish'))
#             self.out.add(keras.layers.Dropout(dropout_rates[i + 2]))
#
#         # self.model = create_ae_mlp(in_dim, out_dim, hidden_units, dropout_rates, lr)
#
#     def call(self, inputs):
#         x = self.encoder(inputs)
#         x = self.decoder(x)
#
#         return self.model(inputs)
#
