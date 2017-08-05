from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf
import collections


class DualSourceAttentionWrapperState(
        collections.namedtuple("DualSourceAttentionWrapperState",
                               ("state1_time", "state1_alignments", "state1_alignment_history", "state2_time", "state2_alignments", "state2_alignment_history", "cell_state",
                                "attention"))):
    pass


def _caluculate_context(attention_wrapper, cell_output, alignments):

    alignments = attention_wrapper._attention_mechanism(
        cell_output, previous_alignments=alignments)

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
        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: array_ops.concat([inputs, attention], -1))
        if name is None:
            name = "dual_source_attention_wrapper"
        name1 = name + "1"
        name2 = name + "2"
        self._attention_layer = layers_core.Dense(
            attention_layer_size, name="attention_layer", use_bias=False)
        self._attention_size = attention_layer_size
        self._cell_input_fn = cell_input_fn
        self._cell = cell
        self._attention_mechanism1 = attention_mechanism1
        self._attention_mechanism2 = attention_mechanism2
        self._attention_wrapper1 = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism1,
            attention_layer_size=None,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            output_attention=output_attention,
            initial_cell_state=initial_cell_state,
            name=name1)
        self._attention_wrapper2 = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism2,
            attention_layer_size=None,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            output_attention=output_attention,
            initial_cell_state=initial_cell_state,
            name=name2)

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
                self._attention_wrapper1, cell_output, state.state1_alignments)
            context2, alignments2 = _caluculate_context(
                self._attention_wrapper2, cell_output, state.state2_alignments)

            attention = self._attention_layer(
                array_ops.concat([cell_output, context1, context2], axis=1))

            next_state = DualSourceAttentionWrapperState(
                state1_time=state.state1_time + 1,
                state1_alignments=alignments1,
                state1_alignment_history=(),
                state2_time=state.state2_time + 1,
                state2_alignments=alignments2,
                state2_alignment_history=(),
                cell_state=next_cell_state,
                attention=attention)

            return attention, next_state
    

    @property
    def output_size(self):
        return self._attention_size


    @property
    def state_size(self):
        return DualSourceAttentionWrapperState(
            state1_time=tensor_shape.TensorShape([]),
            state1_alignments=self._attention_mechanism1.alignments_size,
            state1_alignment_history=(),
            state2_time=tensor_shape.TensorShape([]),
            state2_alignments=self._attention_mechanism2.alignments_size,
            state2_alignment_history=(),
            cell_state=self._cell.state_size,
            attention=self._attention_size)  # alignment_history is sometimes a TensorArray