# A Clockwork RNN #

This repository contains a high-level implementation of the Clockwork-RNN model (CW-RNN, see [[1]](https://arxiv.org/abs/1402.3511)). 

The `ClockworkRNN` class constructs a CW-RNN using Keras Functional API by "unrolling" the DAG graph of the model, instead of computing its block-diagonal matrix representation. This allows the user to use any kind of RNN layer within the CW-RNN.

## Basic usage

For example, to construct a CW-RNN that has in input an audio signal and you want to train the network to predict the next audio sample, you could use the following snippet
```python
from keras.models import Sequential
from cwrnn import ClockworkRNN

model = Sequential()
model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128],
                       units_per_period=8, 
                       input_shape=(None, 1), 
                       output_units=1))
model.compile(optimizer='adam', loss='mse')
model.summary()
```
which produces the following output
```
Layer (type)                 Output Shape              Param #   
=================================================================
clockwork_simple_rnn_1 (Cloc (None, 1)                 2497      
=================================================================
Total params: 2,497
Trainable params: 2,497
Non-trainable params: 0
```

This model uses `SimpleRNN`s as internal layers (by default, as in the original paper), each one with 8 recurrent units. If you are using the TensorFlow backend and you want to train the model on GPU, you can use the fast `SimpleRNN` implementation backed by [CuDNN](https://developer.nvidia.com/cudnn) that can be found in [`cudnnrnn.py`](https://github.com/flandolfi/clockwork-rnn/blob/master/cudnnrnn.py), and use it as internal layer as following
```python
from cudnnrnn import CuDNNSimpleRNN

model = Sequential()
model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128], 
                       units_per_period=8,
                       input_shape=(None, 1), 
                       output_units=1, 
                       rnn_dtype=CuDNNSimpleRNN))
model.compile(optimizer='adam', loss='mse')
model.summary()
```
which produces
```
Layer (type)                 Output Shape              Param #   
=================================================================
clockwork_cu_dnn_simple_rnn_ (None, 1)                 2497      
=================================================================
Total params: 2,497
Trainable params: 2,497
Non-trainable params: 0
```

If you want to use any other Keras' recurrent layer instead, you can just pass its class name to the `rnn_dtype` parameter, as in the next example
```python
model = Sequential()
model.add(ClockworkRNN(periods=[1, 2, 4, 8, 16, 32, 64, 128], 
                       units_per_period=8, 
                       input_shape=(None, 1), 
                       output_units=1, 
                       rnn_dtype='CuDNNLSTM'))
model.compile(optimizer='adam', loss='mse')
model.summary()
```
which produces the following output
```
Layer (type)                 Output Shape              Param #   
=================================================================
clockwork_cu_dnnlstm_1 (Cloc (None, 1)                 10049     
=================================================================
Total params: 10,049
Trainable params: 10,049
Non-trainable params: 0
```

See the code for a more detailed description of the parameters.


## References ##

[1] Koutnik, J., Greff, K., Gomez, F. and Schmidhuber, J., 2014. A clockwork rnn. arXiv preprint [arXiv:1402.3511](https://arxiv.org/abs/1402.3511).