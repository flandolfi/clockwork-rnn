import numpy as np
import keras.backend as K
from keras import layers
from keras.layers import (Concatenate, Dense, Lambda, Layer, MaxPooling1D, 
                          UpSampling1D, TimeDistributed)


class ClockworkRNN(Layer):
    """Clockwork RNN ([Koutnik et al., 2014](https://arxiv.org/abs/1402.3511)).

    Constructs a CW-RNN from RNNs of a given type.

    # Arguments
        periods: List of positive integers. The periods of each internal RNN. 
        units_per_period: Positive integer or list of positive integers.
            Number of units for each internal RNN. If list, it must have the
            same length as `periods`.
        input_shape: Shape of the input data.
        output_units: Positive integer. Dimensionality of the output space.
        output_activation: String or callable. Activation function to use. If
            you don't specify anything, no activation is applied (i.e.,
            "linear" activation: `a(x) = x`). 
        return_sequences: Boolean (default False). Whether to return the last
            output in the output sequence, or the full sequence.
        sort_ascending: Boolean (default False). Whether to sort the periods
            in ascending or descending order (default, as in the original
            paper).
        include_top: Whether to include the fully-connected layer at the top
            of the network.
        dense_kwargs: Dictionary. Optional arguments for the trailing Dense 
            unit (`activation` and `units` keys will be ignored).
        rnn_dtype: The type of RNN to use as clockwork layer. Can be a string
            ("SimpleRNN", "GRU", "LSTM", "CuDNNGRU", "CuDNNLSTM") or any RNN 
            subclass.
        rnn_kwargs: Dictionary. Optional arguments for the internal RNNs 
            (`return_sequences` and `return_state` will be ignored).
    
    """
    def __init__(self, periods, 
                 units_per_period, 
                 output_units,
                 output_activtion='linear',
                 return_sequences=False,
                 sort_ascending=False,
                 include_top=True,
                 dense_kwargs=None,
                 rnn_dtype="SimpleRNN",
                 rnn_kwargs=None,
                 **kwargs):
        if type(rnn_dtype) is str:
            self.rnn_dtype = getattr(layers, rnn_dtype) 
        else:
            self.rnn_dtype = rnn_dtype
        
        ClockworkRNN.__name__ = "Clockwork" + self.rnn_dtype.__name__
        super(ClockworkRNN, self).__init__(**kwargs)

        if type(units_per_period) is list:
            self.units_per_period = units_per_period
        else:
            self.units_per_period = [units_per_period] * len(periods)

        self.periods = periods
        self.rnn_kwargs = rnn_kwargs or {}
        self.rnn_kwargs['return_sequences'] = True
        self.rnn_kwargs['return_state'] = False
        self.rnn_kwargs.pop("units", True)
        self.dense_kwargs = dense_kwargs or {}
        self.dense_kwargs['activation'] = output_activtion
        self.dense_kwargs['units'] = output_units
        self.include_top = include_top
        self.return_sequences = return_sequences
        self.sort_ascending = sort_ascending
        self.blocks = []

    def build(self, input_shape):
        last_shape = input_shape
        output_shapes = []
        
        for period, units in sorted(zip(self.periods, self.units_per_period),
                                    reverse=not self.sort_ascending):
            block, output_shape, last_shape = self._build_clockwork_block(
                units, period, last_shape)
            output_shapes.append(output_shape)
            self.blocks.append(block)

        self.concat_all = Concatenate()
        self.concat_all.build(output_shapes)
        last_shape = self.concat_all.compute_output_shape(output_shapes)

        if not self.return_sequences:
            self.lambda_last = Lambda(lambda x: x[:, -1])
            self.lambda_last.build(last_shape)
            last_shape = self.lambda_last.compute_output_shape(last_shape)

        if self.include_top:
            if self.return_sequences:
                self.dense = TimeDistributed(Dense(**self.dense_kwargs))
            else:
                self.dense = Dense(**self.dense_kwargs)
                
            self.dense.build(last_shape)
            self._trainable_weights.extend(self.dense.trainable_weights)
            last_shape = self.dense.compute_output_shape(last_shape)
                
        super(ClockworkRNN, self).build(input_shape)

    def call(self, x):
        rnns = []
        to_next_block = x

        for block in self.blocks:
            to_dense, to_next_block = self._call_clockwork_block(
                to_next_block, *block)
            rnns.append(to_dense)

        out = self.concat_all(rnns)

        if not self.return_sequences:
            out = self.lambda_last(out)

        if self.include_top:
            out = self.dense(out)
    
        return out

    def compute_output_shape(self, input_shape):
        if self.include_top:
            out_dim = self.dense_kwargs['units']
        else:
            out_dim = np.sum(self.units_per_period)

        if self.return_sequences:
            return input_shape[:-1] + (out_dim,)
        else:
            return input_shape[:-2] + (out_dim,)

    def _delay(self, x):
        return K.temporal_padding(x, (1, 0))[:, :-1]
    
    def _crop(self, x, timesteps):
        return x[:, :K.cast(timesteps, "int32")]

    def _build_clockwork_block(self, units, period, input_shape):
        output_shape = input_shape[:-1] + (units,)
        pool = MaxPooling1D(1, period)
        rnn = self.rnn_dtype(units=units, **self.rnn_kwargs)
        unpool = UpSampling1D(period)
        crop = Lambda(lambda x: self._crop(x[0], x[1]), 
                      output_shape=output_shape[1:])
        delay = Lambda(lambda x: self._delay(x), 
                       output_shape=output_shape[1:])
        concat = Concatenate()
        
        block = (pool, rnn, unpool, crop, delay, concat)
        
        pool.build(input_shape)
        pool_output_shape = pool.compute_output_shape(input_shape)
        rnn.build(pool_output_shape)
        self._trainable_weights.extend(rnn.trainable_weights)
        rnn_output_shape = rnn.compute_output_shape(pool_output_shape)
        unpool.build(rnn_output_shape)
        crop.build([unpool.compute_output_shape(rnn_output_shape), ()])
        delay.build(output_shape)
        concat.build([input_shape, output_shape])

        return block, output_shape, \
            concat.compute_output_shape([input_shape, output_shape])

    def _call_clockwork_block(self, x, pool, rnn, unpool, crop, delay, concat):
        pooled = pool(x)
        rnn_out = rnn(pooled)
        unpooled = unpool(rnn_out)
        to_dense = crop([unpooled, K.shape(x)[1]])
        delayed = delay(to_dense)
        to_next_block = concat([x, delayed])

        return to_dense, to_next_block

    def get_config(self):
        config = super(ClockworkRNN, self).get_config()
        
        config['units_per_period'] = self.units_per_period
        config['periods'] = self.periods
        config['rnn_dtype'] = self.rnn_dtype.__name__
        config['rnn_kwargs'] = self.rnn_kwargs
        config['dense_kwargs'] = self.dense_kwargs
        config['include_top'] = self.include_top
        config['return_sequences'] = self.return_sequences
        config['sort_ascending'] = self.sort_ascending

        return config
