from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import collections


class DualSourceAttentionWrapperState(
        collections.namedtuple("DualSourceAttentionWrapperState",
                               ("state1", "state2", "cell_state",
                                "attention"))):
    pass


def _caluculate_context(attention_wrapper, cell_output, state):

    alignments = attention_wrapper.attention_mechanism(
        cell_output, previous_alignments=state.alignments)

    expanded_alignments = array_ops.expand_dims(alignments, 1)
    attention_mechanism_values = attention_wrapper._attention_mechanism.values
    context = math_ops.matmul(expanded_alignments, attention_mechanism_values)
    context = array_ops.squeeze(context, [1])

    return context, alignments


class DualSourceAttentionWrapper(rnn_cell_impl.RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism1,
                 attention_mechanism2,
                 attention_layer_size,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        super(DualSourceAttentionWrapper, self).__init__(name=name)
        name1 = name + "1"
        name2 = name + "2"
        self._attention_layer = layers_core.Dense(
            attention_layer_size, name="attention_layer", use_bias=False)
        self._cell_input_fn = cell_input_fn
        self._cell = cell
        self._attention_mechanism1 = attention_mechanism1
        self._attention_mechanism2 = attention_mechanism2
        self._attention_wrapper1 = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism1,
            attention_layer_size=None,
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name1)
        self._attention_wrapper2 = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism2,
            attention_layer_size=None,
            alignment_history,
            cell_input_fn,
            output_attention,
            initial_cell_state,
            name2)

    def call(self, inputs, state):
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (cell_output.shape[0].value or
                           array_ops.shape(cell_output)[0])
        with ops.control_dependencies([
                check_ops.assert_equal(
                    cell_batch_size,
                    self._attention_mechanism1.batch_size,
                    message="Non-matching batch sizes."),
                check_ops.assert_equal(
                    cell_batch_size,
                    self._attention_mechanism2.batch_size,
                    message="Non-matching batch sizes.")
        ]):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

            context1, alignments1 = _caluculate_context(
                self._attention_wrapper1, cell_output, state.state1)
            context2, alignments2 = _caluculate_context(
                self._attention_wrapper2, cell_output, state.state2)

            attention = self._attention_layer(
                array_ops.concat([cell_output, context1, context2], axis=1))

            next_state1 = tf.contrib.seq2seq.AttentionWrapperState(
                time=state1.time + 1,
                cell_state=None,
                attention=None,
                alignments=alignments1,
                alignment_history=())
            next_state2 = tf.contrib.seq2seq.AttentionWrapperState(
                time=state2.time + 1,
                cell_state=None,
                attention=None,
                alignments=alignments2,
                alignment_history=())
            next_state = DualSourceAttentionWrapperState(
                state1=next_state1,
                state2=next_state2,
                cell_state=next_cell_state,
                attention=attention)

            return attention, next_state