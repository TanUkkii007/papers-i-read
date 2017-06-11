
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet model.
Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf
import six


from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                    'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                    )


class ResNet(object):
    """ResNet model"""

    def __init__(self, hps, images, labels, mode):
        """
        Args:
            hps: Hyperparameters.
            images: Batches of images. [batch_size, image_size, image_size, 3]
            labels: Batches of labels. [batch_size, num_classes]
            mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []
    
    def build_graph(self):
        """Build a whole graph for the model."""
        # https://www.tensorflow.org/api_docs/python/tf/train/get_or_create_global_step
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()
    
    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope('init'):
            x = self._images
            x = self._conv('init_conv', x, 3, 3, 16, self._stride_arr(1))
        
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        if self.hps.use_bottleneck:
            pass
        else:
            res_func = self._residual
            filters = [16, 16, 32, 64]
        
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activate_before_residual[0])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activate_before_residual[1])
        for i in six.moves.range(1, self.hps.num_residual_units):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(strides[2]), activate_before_residual[2])

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x)
            x = self._global_avg_pool(x)
        
        with tf.variable_scope('logits'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)
        
        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)
    
    def _build_train_op(self):
        pass

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(x, [0,1,2], name='moments')
                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                variance = tf.get_variable('moving_variance', params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y
    
    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers"""
        if activate_before_residual:
            pass
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x)
        
        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)
        
        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x)
            x = self._conv('conv2', x, filter_size=3, in_filters=out_filter, out_filters=out_filter, strides=[1,1,1,1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0,0], [0,0], [0,0],
                    [(out_filter - in_filter) // 2, (out_filter-in_filter) // 2]])
            x += orig_x
        tf.logging.debug('image after unit %s', x.get_shape())
        return x
    
    def _relu(self, x, leakiness=0.0):
        """Relu"""
        # leaky ReLU is formulated as f(x)=max(x, leakiness * x)
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    
    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n))
            )
            return tf.nn.conv2d(x, filter=kernel, strides=strides, padding='SAME')
    
    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1,2])