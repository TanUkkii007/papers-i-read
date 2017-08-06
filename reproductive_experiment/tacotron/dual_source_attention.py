from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
import tensorflow as tf
import collections

_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class DualSourceAttentionWrapperState(
        collections.namedtuple(
            "DualSourceAttentionWrapperState",
            ("state1_time", "state1_alignments", "state1_alignment_history",
             "state2_time", "state2_alignments", "state2_alignment_history",
             "cell_state", "attention"))):
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
                lambda inputs, attention: array_ops.concat([inputs, attention], -1)
            )
        if name is None:
            name = "dual_source_attention_wrapper"
        name1 = name + "1"
        name2 = name + "2"
        self._attention_layer = layers_core.Dense(
            attention_layer_size, name="attention_layer", use_bias=False)
        self._attention_size = attention_layer_size
        self._cell_input_fn = cell_input_fn
        self._alignment_history = alignment_history
        self._cell = cell
        if (initial_cell_state is not None):
            raise NotImplementedError("initial_cell_state is not None")
        else:
            self._initial_cell_state = initial_cell_state
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

            if self._alignment_history:
                alignment_history1 = state.state1_alignment_history.write(
                    state.state1_time, alignments1)
                alignment_history2 = state.state2_alignment_history.write(
                    state.state2_time, alignments2)
            else:
                alignment_history1, alignment_history2 = (), ()

            next_state = DualSourceAttentionWrapperState(
                state1_time=state.state1_time + 1,
                state1_alignments=alignments1,
                state1_alignment_history=alignment_history1,
                state2_time=state.state2_time + 1,
                state2_alignments=alignments2,
                state2_alignment_history=alignment_history2,
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
            attention=self._attention_size
        )  # alignment_history is sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(
                type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)

            with ops.control_dependencies([
                    check_ops.assert_equal(
                        batch_size,
                        self._attention_mechanism1.batch_size,
                        message="Non-matching batch sizes."),
                    check_ops.assert_equal(
                        batch_size,
                        self._attention_mechanism2.batch_size,
                        message="Non-matching batch sizes.")
            ]):

                cell_state = nest.map_structure(
                    lambda s: array_ops.identity(s, name="checked_cell_state"),
                    cell_state)

            if self._alignment_history:
                aligmnent_history1 = tensor_array_ops.TensorArray(
                    dtype=dtype, size=0, dynamic_size=True)
                aligmnent_history2 = tensor_array_ops.TensorArray(
                    dtype=dtype, size=0, dynamic_size=True)
            else:
                aligmnent_history1, aligmnent_history2 = (), ()

            return DualSourceAttentionWrapperState(
                state1_time=array_ops.zeros([], dtype=dtypes.int32),
                state1_alignments=self._attention_mechanism1.
                initial_alignments(batch_size, dtype),
                state1_alignment_history=aligmnent_history1,
                state2_time=array_ops.zeros([], dtype=dtypes.int32),
                state2_alignments=self._attention_mechanism2.
                initial_alignments(batch_size, dtype),
                state2_alignment_history=aligmnent_history2,
                cell_state=cell_state,
                attention=_zero_state_tensors(self._attention_size, batch_size,
                                              dtype))
