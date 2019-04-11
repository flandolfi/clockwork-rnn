# A Clockwork RNN #

This repository contains a high-level implementation of the Clockwork-RNN model (CW-RNN, see [[1]](https://arxiv.org/abs/1402.3511)). 

The `ClockworkRNN` class constructs a CW-RNN using Keras Functional API by "unrolling" the DAG graph of the model, instead of computing its block-diagonal matrix representation. This allows the user to use any kind of RNN layer whithin the CW-RNN, as long as it support masking (e.g., `CuDNNGRU` and `CuDNNLSTM` are not supported yet).

## Basic usage

For example, to construct a CW-RNN that has in input an audio signal and we want to train the network to predict the next audio sample, we could use the following snipped
```python
from keras.models import Sequential
from keras.layers import InputLayer
from cwrnn import ClockworkRNN

model = Sequential()
model.add(InputLayer((None, 1)))
model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128],
                       units_per_period=8, 
                       output_units=1))
model.compile(optimizer='adam', loss='mse')
model.summary()
```
which produces the following output
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
clockwork_SimpleRNN (Clockwo (None, 1)                 2497      
=================================================================
Total params: 2,497
Trainable params: 2,497
Non-trainable params: 0
_________________________________________________________________
```


This model uses `SimpleRNN`s as internal units (by default, as in the original paper), each one with 8 recurrent units. If we want to use an `LSTM` instead, we can just change the `rnn_dtype` parameter, as in the next example
```python
model = Sequential()
model.add(InputLayer((None, 1)))
model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128], 
                       units_per_period=8, 
                       output_units=1, 
                       rnn_dtype='LSTM'))
model.compile(optimizer='adam', loss='mse')
model.summary()
```
which produces the following output
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
clockwork_LSTM (ClockworkRNN (None, 1)                 9793      
=================================================================
Total params: 9,793
Trainable params: 9,793
Non-trainable params: 0
_________________________________________________________________
```

See the code for a more detailed description of the parameters.


## References ##

[1] Koutnik, J., Greff, K., Gomez, F. and Schmidhuber, J., 2014. A clockwork rnn. arXiv preprint [arXiv:1402.3511](https://arxiv.org/abs/1402.3511).