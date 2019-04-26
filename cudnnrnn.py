from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers.cudnn_recurrent import _CuDNNRNN
from keras import initializers
from keras import regularizers
from keras import constraints

from collections import namedtuple


class CuDNNSimpleRNN(_CuDNNRNN):
    """Fast SimpleRNN implementation backed by [CuDNN](https://developer.nvidia.com/cudnn).
    Can only be run on GPU, with the TensorFlow backend.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
    """

    def __init__(self, units,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 stateful=False,
                 **kwargs):
        self.units = units
        super(CuDNNSimpleRNN, self).__init__(
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
            **kwargs)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    @property
    def cell(self):
        Cell = namedtuple('cell', 'state_size')
        cell = Cell(state_size=self.units)
        return cell

    def build(self, input_shape):
        super(CuDNNSimpleRNN, self).build(input_shape)
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        input_dim = input_shape[-1]

        from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
        self._cudnn_rnn = cudnn_rnn_ops.CudnnRNNTanh(
            num_layers=1,
            num_units=self.units,
            input_size=input_dim,
            input_mode='linear_input')

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.bias = self.add_weight(shape=(self.units,),
                                    name='bias',
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.built = True

    def _process_batch(self, inputs, initial_state):
        import tensorflow as tf
        inputs = tf.transpose(inputs, (1, 0, 2))
        input_h = initial_state[0]
        input_h = tf.expand_dims(input_h, axis=0)

        params = self._canonical_to_params(
            weights=[
                self.kernel,
                self.recurrent_kernel,
            ],
            biases=[
                self.bias,
            ],
        )
        outputs, h = self._cudnn_rnn(
            inputs,
            input_h=input_h,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
        if self.return_sequences:
            output = tf.transpose(outputs, (1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h]

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)}
        base_config = super(CuDNNSimpleRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
