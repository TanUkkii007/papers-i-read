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
        with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
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
                score = math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])
            alignments = self._probability_fn(score, previous_alignments)
            return alignments
