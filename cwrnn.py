import numpy as np
import keras.backend as K
from keras.layers import (Concatenate, Dense, Lambda, Masking, Layer,
                          TimeDistributed, recurrent)


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
        mask_value: Float (default 0). Values that will appear in the masked 
            steps of each internal RNN (i.e., the ones that do not satisfy
            `step % period == 0`).
        include_top: Whether to include the fully-connected layer at the top
            of the network.
        dense_kwargs: Dictionary. Optional arguments for the trailing Dense 
            unit (`activation` and `units` keys will be ignored).
        rnn_dtype: The type of RNN to use as clockwork layer. Can be a string
            ("SimpleRNN", "GRU", "LSTM") or any RNN subclass, but must support
            masking (e.g, CuDNNGRU and CuDNNLSTM are not supported).
        rnn_kwargs: Dictionary. Optional arguments for the internal RNNs 
            (`return_sequences` and `return_state` will be ignored).
    
    """
    def __init__(self, periods, 
                 units_per_period, 
                 output_units,
                 output_activtion='linear',
                 return_sequences=False,
                 sort_ascending=False,
                 mask_value=0.,
                 include_top=True,
                 dense_kwargs=None,
                 rnn_dtype="SimpleRNN",
                 rnn_kwargs=None,
                 **kwargs):
        if type(rnn_dtype) is str:
            self.rnn_dtype = getattr(recurrent, rnn_dtype) 
        else:
            self.rnn_dtype = rnn_dtype
        
        if 'name' not in kwargs:
            kwargs['name'] = "Clockwork" + self.rnn_dtype.__name__
        
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
        self.rnn_kwargs.pop("name", True)
        self.dense_kwargs = dense_kwargs or {}
        self.dense_kwargs['activation'] = output_activtion
        self.dense_kwargs['units'] = output_units
        self.dense_kwargs['name'] = 'cw_output'
        self.include_top = include_top
        self.return_sequences = return_sequences
        self.sort_ascending = sort_ascending
        self.mask_value = mask_value
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

        self.concat_all = Concatenate(name='rnn_outputs')
        self.concat_all.build(output_shapes)
        last_shape = self.concat_all.compute_output_shape(output_shapes)

        if not self.return_sequences:
            self.lambda_last = Lambda(lambda x: x[:, -1], name='last_output')
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

    def _apply_mask(self, x, period, value=0):
        # Ugly fix
        if K.ndim(x) < 3 or np.any(np.equal(K.int_shape(x), None)):
            return x

        mask = np.zeros((K.int_shape(x)[-2], 1))
        mask[::period, 0] = 1
        mask = K.constant(mask)

        return x*mask + value*(1 - mask)

    def _delay(self, x):
        return K.temporal_padding(x, (1, 0))[:, :-1]

    def _build_clockwork_block(self, units, period, input_shape):
        lambda_filter = Lambda(lambda x: self._apply_mask(x, period, 
                                                value=self.mask_value),
                                name='filter_at_{}'.format(period))
        mask = Masking(self.mask_value, name='mask_at_{}'.format(period))
        rnn = self.rnn_dtype(units=units, name='rnn_at_{}'.format(period), 
                             **self.rnn_kwargs)
        lambda_delay = Lambda(lambda x: self._delay(x), 
                                name='delay_at_{}'.format(period))
        concat = Concatenate(name='concat_at_{}'.format(period))
        
        block = (lambda_filter, mask, rnn, lambda_delay, concat)
        
        lambda_filter.build(input_shape)
        mask.build(input_shape)
        rnn.build(input_shape)
        self._trainable_weights.extend(rnn.trainable_weights)
        rnn_output_shape = rnn.compute_output_shape(input_shape)
        lambda_delay.build(rnn_output_shape)
        concat.build([input_shape, rnn_output_shape])

        return block, rnn_output_shape, \
            concat.compute_output_shape([input_shape, rnn_output_shape])

    def _call_clockwork_block(self, x, lambda_f, mask, rnn, lambda_d, concat):
        filtered = lambda_f(x)
        masked = mask(filtered)
        to_dense = rnn(masked)
        delayed = lambda_d(to_dense)
        to_next_block = concat([x, delayed])

        return to_dense, to_next_block

    def get_config(self):
        config = super(ClockworkRNN, self).get_config()
        
        config['units_per_period'] = self.units_per_period
        config['periods'] = self.periods
        config['rnn_dtype'] = self.rnn_dtype
        config['rnn_kwargs'] = self.rnn_kwargs
        config['dense_kwargs'] = self.dense_kwargs
        config['include_top'] = self.include_top
        config['return_sequences'] = self.return_sequences
        config['sort_ascending'] = self.sort_ascending
        config['mask_value'] = self.mask_value

        return config
