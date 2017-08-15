'''
https://github.com/keithito/tacotron/blob/master/models/rnn_wrappers.py
'''

from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf


class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    def __init__(self, cell, is_training):
        super(DecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training
    

    @property
    def state_size(self):
        return self._cell.state_size


    @property
    def output_size(self):
        return self._cell.output_size


    def call(self, inputs, state):
        from modules import prenet
        prenet_out = prenet(inputs, self._is_training, scope="decoder_prenet")
        return self._cell(prenet_out, state)


    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.
    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
    

    @property
    def state_size(self):
        return self._cell.state_size


    @property
    def output_size(self):
        # see AttentionWrapper.state_size
        return self._cell.output_size + self._cell.state_size.attention


    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state


    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)