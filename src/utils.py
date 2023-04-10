import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):

    """Create a sample plot for indices of a cross-validation object."""

    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)  # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    # ax.scatter(range(len(X)), [ii + 1.5] * len(X),
    #            c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['day'] # 'target',
    ax.set(yticks=np.arange(n_splits + 1) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits + 1.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax



def IC(y_true, y_pred, sample_weights=None):
    if sample_weights is not None:
        y_true = tf.boolean_mask(y_true, sample_weights)
        y_pred = tf.boolean_mask(y_pred, sample_weights)
    correlation = tfp.stats.correlation(y_true, y_pred, sample_axis=1, event_axis=None)
    return tf.reduce_mean(correlation)

# Custom loss function for MSE + Correlation
@tf.function
def mse_corr_loss(y_true, y_pred, sample_weights=None):
    # if sample_weights is not None:
    #     y_true = tf.boolean_mask(y_true, sample_weights)
    #     y_pred = tf.boolean_mask(y_pred, sample_weights)
    eta = IC(y_true, y_pred, sample_weights)
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred, sample_weights) + tf.constant(0.05)/tf.math.maximum(eta, 0.0001)


class PearsonCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name='pearson_correlation', **kwargs):
        super(PearsonCorrelation, self).__init__(name=name, **kwargs)
        self.correlation = self.add_weight(name='correlation', initializer='zeros')
        self.running_times = self.add_weight(name='running_times', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        rank = len(y_true.shape)
        axis = 0 if rank == 1 else 1

        if sample_weight is None:
            sample_weight = tf.ones_like(y_true)

        # Apply sample weights to y_true and y_pred
        y_true_weighted = y_true * sample_weight
        y_pred_weighted = y_pred * sample_weight

        # Compute the correlation using tfp.stats.correlation
        correlation = tfp.stats.correlation(y_true_weighted, y_pred_weighted, sample_axis=axis, event_axis=None)
        self.correlation.assign(tf.reduce_mean(correlation))
        self.running_times.assign_add(1)

    def result(self):
        return self.correlation/self.running_times

    def reset_states(self):
        self.correlation.assign(0.)
        self.running_times.assign(0.)

class corr_metric(tf.keras.metrics.Metric):
    def __init__(self, name='correlation', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.sum_x = self.add_weight(name='sum_x', initializer='zeros', shape=(1,))
        self.sum_y = self.add_weight(name='sum_y', initializer='zeros', shape=(1,))
        self.sum_xy = self.add_weight(name='sum_xy', initializer='zeros', shape=(1,))
        self.sum_x_squared = self.add_weight(name='sum_x_squared', initializer='zeros', shape=(1,))
        self.sum_y_squared = self.add_weight(name='sum_y_squared', initializer='zeros', shape=(1,))
        self.initial_update = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            y_true = tf.multiply(y_true, sample_weight)
            y_pred = tf.multiply(y_pred, sample_weight)

        if self.initial_update:
            self.sum_x.assign(tf.zeros_like(self.sum_x, shape=tf.shape(y_true)[1:]))
            self.sum_y.assign(tf.zeros_like(self.sum_y, shape=tf.shape(y_pred)[1:]))
            self.sum_xy.assign(tf.zeros_like(self.sum_xy, shape=tf.shape(y_true)[1:]))
            self.sum_x_squared.assign(tf.zeros_like(self.sum_x_squared, shape=tf.shape(y_true)[1:]))
            self.sum_y_squared.assign(tf.zeros_like(self.sum_y_squared, shape=tf.shape(y_pred)[1:]))
            self.initial_update = False

        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.sum_x.assign_add(tf.reduce_sum(y_true, axis=0))
        self.sum_y.assign_add(tf.reduce_sum(y_pred, axis=0))
        self.sum_xy.assign_add(tf.reduce_sum(tf.multiply(y_true, y_pred), axis=0))
        self.sum_x_squared.assign_add(tf.reduce_sum(tf.square(y_true), axis=0))
        self.sum_y_squared.assign_add(tf.reduce_sum(tf.square(y_pred), axis=0))

    def result(self):
        num = self.sum_xy - (self.sum_x * self.sum_y) / self.total_samples
        den = tf.sqrt((self.sum_x_squared - tf.square(self.sum_x) / self.total_samples) * (
                    self.sum_y_squared - tf.square(self.sum_y) / self.total_samples))
        return tf.reduce_mean(num / den)

    def reset_state(self):
        self.total_samples.assign(0)
        self.sum_x.assign(tf.zeros_like(self.sum_x))
        self.sum_y.assign(tf.zeros_like(self.sum_y))
        self.sum_xy.assign(tf.zeros_like(self.sum_xy))
        self.sum_x_squared.assign(tf.zeros_like(self.sum_x_squared))
        self.sum_y_squared.assign(tf.zeros_like(self.sum_y_squared))


