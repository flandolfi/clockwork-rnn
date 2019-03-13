import keras.backend as K
import numpy as np
from keras import Model, layers
from keras.layers import (Concatenate, Dense, Input, Lambda, Masking,
                          TimeDistributed, SimpleRNN, LSTM, GRU, RNN)


def clockwork(dtype : RNN):
    """Clockwork class decorator.

    Returns a Clockwork-RNN (CW-RNN) model using the given RNN sub-class as 
    inner recurrent layers.

    # Arguments
        dtype: RNN sub-class. The RNN type that will be used inside the 
            CW-RNN. Must support masking (e.g., CuDNNRNN are not supported
            yet).
    
    # Returns
        The decorated class.
    """

    class Clockwork(Model):
        """Clockwork RNN 
        ([Koutnik et al., 2014](https://arxiv.org/abs/1402.3511))

        Constructs a CW-RNN from RNNs of a given type.

        # Arguments
            periods: List of positive integers. The periods of each internal
                RNN. 
            units_per_period: Positive integer or list of positive integers.
                Number of units for each internal RNN. If list, it must have
                the same length as `periods`.
            input_shape: Shape of the input data.
            output_units: Positive integer. Dimensionality of the output 
                space.
            output_activation: String or callable. Activation function to use.
                If you don't specify anything, no activation is applied (i.e.,
                "linear" activation: `a(x) = x`). 
            return_sequences: Boolean (default False). Whether to return the 
                last output in the output sequence, or the full sequence.
            sort_ascending: Boolean (default True). Whether to sort the 
                periods in ascending or descending order (default, as in the
                original paper).
            mask_value: Float (default 0). Values that will appear in the
                masked steps of each internal RNN (i.e., the ones that do not
                satisfy `step % period == 0`).
            include_top: Whether to include the fully-connected layer at the
                top of the network.
            dense_kwargs: Dictionary. Optional arguments for the trailing
                Dense unit (`activation` and `units` keys will be ignored).
            rnn_kwargs: Dictionary. Optional arguments for the internal RNNs 
                (`return_sequences` and `return_state` will be ignored).
        """

        def __init__(self, periods, 
                     units_per_period, 
                     input_shape, 
                     output_units,
                     output_activtion='linear',
                     return_sequences=False,
                     sort_ascending=False,
                     mask_value=0.,
                     include_top=True,
                     dense_kwargs=None,
                     **rnn_kwargs):
            if type(units_per_period) is list:
                self.units_per_period = units_per_period
            else:
                self.units_per_period = [units_per_period] * len(periods)

            self.periods = periods
            self.rnn_kwargs = rnn_kwargs or {}
            self.rnn_kwargs['return_sequences'] = True
            self.rnn_kwargs['return_state'] = False
            self.dense_kwargs = dense_kwargs or {}
            self.dense_kwargs['activation'] = output_activtion
            self.dense_kwargs['units'] = output_units
            self.input_layer = Input(input_shape, name='cw_input')
            self.mask_value = mask_value

            rnns = []
            last_output = self.input_layer

            for period, units in sorted(zip(self.periods, 
                                            self.units_per_period),
                                        reverse=not sort_ascending,
                                        key=lambda t: t[0]):
                to_dense, to_next_block = self._build_clockwork_block(
                    last_output, units, period)
                rnns.append(to_dense)
                last_output = to_next_block

            concat = Concatenate(name='rnn_outputs')(rnns)

            if not include_top:
                self.output_layer = concat
            elif return_sequences:
                self.output_layer = TimeDistributed(
                    Dense(**self.dense_kwargs, name='cw_output'))(concat)
            else:
                last = Lambda(lambda x: x[:, -1], name='last_output')(concat)
                self.output_layer = Dense(**self.dense_kwargs, 
                                          name='cw_output')(last)

            super(Clockwork, self).__init__(name=dtype.__name__,
                                            inputs=self.input_layer,
                                            outputs=self.output_layer)

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

        def _build_clockwork_block(self, x, units, period):
            mask = Lambda(lambda x: self._apply_mask(x, period, 
                                                     value=self.mask_value),
                          name='filter_at_{}'.format(period))(x)
            mask = Masking(self.mask_value,
                           name='mask_at_{}'.format(period))(mask)
            hidden = dtype(units=units,
                           name='rnn_at_{}'.format(period),
                           **self.rnn_kwargs)(mask)
            delayed = Lambda(lambda x: self._delay(x),
                             name='delay_at_{}'.format(period))(hidden)
            delayed = Concatenate(
                name='concat_at_{}'.format(period))([x, delayed])

            return hidden, delayed

    return Clockwork


@clockwork
class ClockworkRNN(SimpleRNN):
    pass


@clockwork
class ClockworkGRU(GRU):
    pass


@clockwork
class ClockworkLSTM(LSTM):
    pass
