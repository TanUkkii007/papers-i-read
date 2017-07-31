# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A powerful dynamic attention wrapper object."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf


# from rnn_cell_impl.py
def concat(prefix, suffix, static=False):
    """Concat that enables int, Tensor, or TensorShape values.

    This function takes a size specification, which can be an integer, a
    TensorShape, or a Tensor, and converts it into a concatenated Tensor
    (if static = False) or a list of integers (if static = True).

    Args:
        prefix: The prefix; usually the batch size (and/or time step size).
        (TensorShape, int, or Tensor.)
        suffix: TensorShape, int, or Tensor.
        static: If `True`, return a python list with possibly unknown dimensions.
        Otherwise return a `Tensor`.

    Returns:
        shape: the concatenation of prefix and suffix.

    Raises:
        ValueError: if `suffix` is not a scalar or vector (or TensorShape).
        ValueError: if prefix or suffix was `None` and asked for dynamic
        Tensors out.
    """
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    # constant_op.constant == tf.constant
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)

    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    # constant_op.constant == tf.constant
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

    if static:
        shape = tensor_shape.as_shape(p_static).concatenate(s_static)
        shape = shape.as_list() if shape.ndims is not None else None
    else:
        if p is None or s is None:
            raise ValueError("Provided a prefix or suffix of None: %s and %s" %
                             (prefix, suffix))
        shape = array_ops.concat((p, s), 0)
    return shape


# from rnn_cell_impl.py
def zero_state_tensors(state_size, batch_size, dtype):
    """
    Create tensors of zeros based on state_size, batch_size, and dtype.
    Args:
        state_size: int32 scalar
        batch_size: int32 scalar
    """

    def get_state_shape(s):
        """Combine s with batch_size to get a proper tensor shape."""
        c = concat(batch_size, s)
        c_static = concat(batch_size, s, static=True)
        size = array_ops.zeros(c, dtype=dtype)
        size.set_shape(c_static)
        return size

    return nest.map_structure(get_state_shape, state_size)


def prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
    """Convert to tensor and possibly mask `memory`.
  
    Args:
        memory: `Tensor`, shaped `[batch_size, max_time, ...]`.
        memory_sequence_length: `int32` `Tensor`, shaped `[batch_size]`.
        check_inner_dims_defined: Python boolean.  If `True`, the `memory`
        argument's shape is checked to ensure all but the two outermost
        dimensions are fully defined.
  
    Returns:
        A (possibly masked), checked, new `memory`.

    Raises:
        ValueError: If `check_inner_dims_defined` is `True` and not
        `memory.shape[2:].is_fully_defined()`.
    """
    pass


def maybe_mask_score(score, memory_sequence_length, score_mask_value):
    if memory_sequence_length is None:
        return score
    with ops.control_dependencies([
            check_ops.assert_positive(
                memory_sequence_length,
                message=
                ("All values in memory_sequence_length must greater than zero."
                 ))
    ]):
        score_mask = array_ops.sequence_mask(
            memory_sequence_length, maxlen=array_ops.shape(score)[1])
        score_mask_values = score_mask_value * array_ops.ones_like(score)
        return array_ops.where(score_mask, score, score_mask_values)


class AttentionMechanism(object):
    pass


class BaseAttentionMechanism(AttentionMechanism):
    """A base AttentionMechanism class providing common functionality.

    Common functionality includes:
        1. Storing the query and memory layers.
        2. Preprocessing and storing the memory.
    """

    def __init__(self,
                 query_layer,
                 memory,
                 probability_fn,
                 memory_sequence_length=None,
                 memory_layer=None,
                 check_inner_dims_defined=True,
                 score_mask_value=float("-inf"),
                 name=None):
        """Construct base AttentionMechanism class.
        Args:
            query_layer: Callable.  Instance of `tf.layers.Layer`.  The layer's depth
                must match the depth of `memory_layer`.  If `query_layer` is not
                provided, the shape of `query` must match that of `memory_layer`.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            probability_fn: A `callable`.  Converts the score and previous alignments
                to probabilities. Its signature should be:
                `probabilities = probability_fn(score, previous_alignments)`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            memory_layer: Instance of `tf.layers.Layer` (may be None).  The layer's
                depth must match the depth of `query_layer`.
                If `memory_layer` is not provided, the shape of `memory` must match
                that of `query_layer`.
            check_inner_dims_defined: Python boolean.  If `True`, the `memory`
                argument's shape is checked to ensure all but the two outermost
                dimensions are fully defined.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        """
        self._query_layer = query_layer
        self._memory_layer = memory_layer
        self._probability_fn = lambda score, prev: (probability_fn(maybe_mask_score(score, memory_sequence_length, score_mask_value), prev))
        with ops.name_scope(name, "BaseAttentionMechanismInit",
                            nest.flatten(memory)):
            self._values = prepare_memory(memory, memory_sequence_length,
                                          check_inner_dims_defined)
            self._keys = (
                self.memory_layer(self._values)  # pylint: disable=not-callable
                if self.memory_layer else self._values)
            self._batch_size = (self._keys.shape[0].value or
                                array_ops.shape(self._keys)[0])
            self._alignments_size = (self._keys.shape[1].value or
                                     array_ops.shape(self._keys)[1])

    @property
    def memory_layer(self):
        return self._memory_layer

    @property
    def query_layer(self):
        return self._query_layer

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    def initial_alignments(self, batch_size, dtype):
        """Creates the initial alignment values for the `AttentionWrapper` class.
        
        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).
        
        The default behavior is to return a tensor of all zeros.
        
        Args:
            batch_size: `int32` scalar, the batch_size.
            dtype: The `dtype`.
        
        Returns:
            A `dtype` tensor shaped `[batch_size, alignments_size]`
            (`alignments_size` is the values' `max_time`).
        """
        max_time = self._alignments_size
        return zero_state_tensors(max_time, batch_size, dtype)


class BahdanauAttention(BaseAttentionMechanism):
    """Implements Bhadanau-style (additive) attention.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=float("-inf"),
                 name="BahdanauAttention"):
        """Construct the Attention mechanism.

        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is @{tf.nn.softmax}. Other options include
                @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            name: Name to use when creating ops.
        """
        if probability_fn is None:
            probability_fn = nn_ops.softmax
        wrapped_probability_fn = lambda score, _: probability_fn(score)
        super(BahdanauAttention, self).__init__(
            query_layer=layers_core.Dense(
                num_units, name="query_layer", use_bias=False),
            memory_layer=layers_core.Dense(
                num_units, name="memory_layer", use_bias=False),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, previous_alignments):
        """Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            previous_alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).

        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        with variable_scope.variable_scope(None, "bahdanau_attention",
                                           [query]):
            processed_query = self.query_layer(
                query) if self.query_layer else query
            dtype = processed_query.dtype
            # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
            processed_query = array_ops.expand_dims(processed_query, 1)
            keys = self._keys
            v = variable_scope.get_variable(
                "attention_v", [self._num_units], dtype=dtype)
            if self._normalize:
                pass
            else:
                '''
                attention score formula
                $\alpha_{ij} = \mathrm{align}(\mathrm{he}_j, \mathrm{ha}_i) = \frac{\exp(\mathrm{score}(\mathrm{he}_j, \mathrm{ha}_i))}{\sum_{j}\exp(\mathrm{score}(\mathrm{he}_j, \mathrm{ha}_i))}$
                alignment model formula (concat method variant)
                $\mathrm{score}(\mathrm{he}_j, \mathrm{ha}_i) = v^\top \tanh(W\mathrm{he}_j + U\mathrm{ha}_i)$
                '''
                score = math_ops.reduce_sum(
                    v * math_ops.tanh(keys + processed_query), [2])
            alignments = self._probability_fn(score, previous_alignments)
            return alignments


class AttentionWrapper(rnn_cell_impl.RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        """Construct the `AttentionWrapper`.

        Args:
            cell: An instance of `RNNCell`.
            attention_mechanism: An instance of `AttentionMechanism`.
            attention_layer_size: Python integer, the depth of the attention (output)
                layer. If None (default), use the context as attention at each time
                step. Otherwise, feed the context and cell output into the attention
                layer to generate attention at each time step.
            alignment_history: Python boolean, whether to store alignment history
                from all time steps in the final output state (currently stored as a
                time major `TensorArray` on which you must call `stack()`).
            cell_input_fn: (optional) A `callable`.  The default is:
                `lambda inputs, attention: array_ops.concat([inputs, attention], -1)`.
            output_attention: Python bool.  If `True` (default), the output at each
                time step is the attention value.  This is the behavior of Luong-style
                attention mechanisms.  If `False`, the output at each time step is
                the output of `cell`.  This is the beahvior of Bhadanau-style
                attention mechanisms.  In both cases, the `attention` tensor is
                propagated to the next time step via the state and is used there.
                This flag only controls whether the attention mechanism is propagated
                up to the next cell in an RNN stack or to the top RNN output.
            initial_cell_state: The initial state value to use for the cell when
                the user calls `zero_state()`.  Note that if this value is provided
                now, and the user uses a `batch_size` argument of `zero_state` which
                does not match the batch size of `initial_cell_state`, proper
                behavior is not guaranteed.
            name: Name to use when creating ops.
        """
        super(AttentionWrapper, self).__init__(name=name)

        if cell_input_fn is None:
            cell_input_fn = (
                lambda inputs, attention: array_ops.concat([inputs, attention], -1)
            )

        if attention_layer_size is not None:
            self._attention_layer = layers_core.Dense(
                attention_layer_size, name="attention_layer", use_bias=False)
            self._attention_size = attention_layer_size
        else:
            self._attention_layer = None
            self._attention_size = attention_mechanism.values.get_shape()[
                -1].value

        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history

        with ops.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (final_state_tensor.shape[0].value or
                                    array_ops.shape(final_state_tensor)[0])
                with ops.control_dependencies([
                        check_ops.assert_equal(
                            state_batch_size,
                            self._attention_mechanism.batch_size,
                            message="Non-matching batch sizes.")
                ]):
                    self._initial_cell_state = nest.map_structure(
                        lambda s: array_ops.identity(s, name="check_initial_cell_state"),
                        initial_cell_state
                    )

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_size` outputs).

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
              tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `DynamicAttentionWrapperState`
               containing the state calculated at this time step.
        """
        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (cell_output.shape[0].value or
                           array_ops.shape(cell_output)[0])

        with ops.control_dependencies([
                check_ops.assert_equal(
                    cell_batch_size,
                    self._attention_mechanism.batch_size,
                    message="Non-matching batch sizes.")
        ]):
            cell_output = array_ops.identity(
                cell_output, name="checked_cell_output")

        alignments = self._attention_mechanism(
            cell_output, previous_alignments=state.alignments)

        # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
        expanded_alignments = array_ops.expand_dims(alignments, axis=1)
        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # alignments shape is
        #   [batch_size, 1, memory_time]
        # attention_mechanism.values shape is
        #   [batch_size, memory_time, attention_mechanism.num_units]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, attention_mechanism.num_units].
        # we then squeeze out the singleton dim.
        attention_mechanism_values = self._attention_mechanism.values
        context = math_ops.matmul(expanded_alignments,
                                  attention_mechanism_values)
        context = array_ops.squeeze(context, [1])