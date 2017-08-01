from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


def _caluculate_context(attention, inputs, state):
    cell_inputs = attention._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = attention._cell(cell_inputs, cell_state)

    cell_batch_size = (cell_output.shape[0].value or
                       array_ops.shape(cell_output)[0])
    with ops.control_dependencies([
            check_ops.assert_equal(
                cell_batch_size,
                attention.attention_mechanism.batch_size,
                message="Non-matching batch sizes.")
    ]):
        cell_output = array_ops.identity(
            cell_output, name="checked_cell_output")

    alignments = attention.attention_mechanism(
        cell_output, previous_alignments=state.alignments)

    expanded_alignments = array_ops.expand_dims(alignments, 1)
    attention_mechanism_values = attention._attention_mechanism.values
    context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
    context = array_ops.squeeze(context, [1])

    return context, cell_output


class DualSourceAttentionWrapper(rnn_cell_impl.RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism1,
                 attention_mechanism2,
                 attention_layer_size1=None,
                 attention_layer_size2=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        super(DualSourceAttentionWrapper, self).__init__(name=name)
        name1 = name + "1"
        name2 = name + "2"
        self._attention_wrapper1 = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism1, attention_layer_size1,
            alignment_history, cell_input_fn, output_attention,
            initial_cell_state, name1)
        self._attention_wrapper2 = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism2, attention_layer_size2,
            alignment_history, cell_input_fn, output_attention,
            initial_cell_state, name2)

    def call(self, inputs, state):
        context1, cell_output1 = _caluculate_context(self._attention_wrapper1,
                                                     inputs, state)
        context2, cell_output2 = _caluculate_context(self._attention_wrapper2,
                                                     inputs, state)

        if self._attention_wrapper1._attention_layer is not None:
            attention1 = self._attention_wrapper1._attention_layer(
                array_ops.concat([cell_output1, context1], axis=1))
