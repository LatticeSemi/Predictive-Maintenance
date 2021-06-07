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
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResNet(object):
    """ResNet model."""

    def __init__(self, hps, images, labels, mode):
        """ResNet constructor.

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
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _build_model(self):
        """Build the core model within the graph."""
        depth = [8, 8, 16, 16, 16, 22, 24]

        bias_on = False

        if self.mode == 'train':
            phase_train = True
        else:
            phase_train = False
        fire1 = self._vgg_layer('fire1', self._images, oc=depth[0], freeze=False, pool_en=True,
                                bias_on=bias_on, phase_train=phase_train)
        fire2 = self._vgg_layer('fire2', fire1, oc=depth[1], freeze=False, pool_en=False,
                                bias_on=bias_on, phase_train=phase_train)
        fire3 = self._vgg_layer('fire3', fire2, oc=depth[2], freeze=False, pool_en=True,
                                bias_on=bias_on, phase_train=phase_train)
        fire4 = self._vgg_layer('fire4', fire3, oc=depth[3], freeze=False, pool_en=False,
                                bias_on=bias_on, phase_train=phase_train)
        fire5 = self._vgg_layer('fire5', fire4, oc=depth[4], freeze=False, pool_en=True,
                                bias_on=bias_on, phase_train=phase_train)
        fire6 = self._vgg_layer('fire6', fire5, oc=depth[5], freeze=False, pool_en=False,
                                bias_on=bias_on, phase_train=phase_train)
        fire7 = self._vgg_layer('fire7', fire6, oc=depth[6], freeze=False, pool_en=True,
                                bias_on=bias_on, phase_train=phase_train)
        if phase_train:
            fire_o = tf.nn.dropout(fire7, 0.8)
        else:
            fire_o = tf.nn.dropout(fire7, 1)
        logits = self._fc_layer('logit', fire_o, self.hps.num_classes, flatten=True, relu=False, xavier=True)

        self.fire1 = fire1
        self.fire2 = fire2
        self.fire3 = fire3
        self.fire4 = fire4
        self.fire5 = fire5
        self.fire6 = fire6
        self.logits = logits

        self.predictions = tf.nn.softmax(logits)

        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)
        self.confusion_matrix = tf.confusion_matrix(tf.argmax(self.labels, 1), tf.argmax(self.predictions, 1), num_classes=self.hps.num_classes)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()

        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        elif self.hps.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.lrn_rate)
        elif self.hps.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.lrn_rate, decay=0.9, momentum=0.9, epsilon=1.0)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    @staticmethod
    def _batch_norm_tensor2(name, x, phase_train=True):  # works well for phase_train tensor
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))
            tf.summary.histogram('bn_gamma', gamma)
            tf.summary.histogram('bn_beta', beta)

            if phase_train:
                mean_train, variance_train = tf.nn.moments(x, [0, 1, 2], name='moments')

                moving_mean = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean_train, 0.9)
                update_moving_var = moving_averages.assign_moving_average(moving_variance, variance_train, 0.9)
                control_inputs_train = [update_moving_mean, update_moving_var]
                with tf.control_dependencies(control_inputs_train):
                    mean, variance = tf.identity(mean_train), tf.identity(variance_train)  # , [control_inputs_train]

            else:
                mean_eval = tf.get_variable(
                    'moving_mean', params_shape, tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                variance_eval = tf.get_variable(
                    'moving_variance', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
                mean, variance = mean_eval, variance_eval

            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    @staticmethod
    def lin_8b_quant(w, min_rng=-0.5, max_rng=0.5):
        min_clip = tf.rint(min_rng * 256 / (max_rng - min_rng))
        max_clip = tf.rint(max_rng * 256 / (max_rng - min_rng)) - 1  # 127, 255

        wq = 256.0 * w / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
        wq = tf.rint(wq)  # integer (quantization)
        wq = tf.clip_by_value(wq, min_clip, max_clip)  # fit into 256 linear quantization
        wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
        wclip = tf.clip_by_value(w, min_rng, max_rng)  # linear value w/ clipping
        return wclip + tf.stop_gradient(wq - wclip)

    def binary_wrapper(self, x, a_bin=16, min_rng=-0.5, max_rng=0.5):  # activation binarization
        if a_bin == 8 and False:
            x_quant = self.lin_8b_quant(x, min_rng=min_rng, max_rng=max_rng)
            return tf.nn.relu(x_quant)
        else:
            return tf.nn.relu(x)

    def _conv_layer(self, layer_name, inputs, filters, size, stride, padding='SAME',
                    freeze=False, xavier=False, relu=True, w_bin=16, bias_on=True, stddev=0.001):
        """Convolutional layer operation constructor.
        Args:
          layer_name: layer name.
          inputs: input tensor
          filters: number of output filters.
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
          freeze: if true, then do not train the parameters in this layer.
          xavier: whether to use xavier weight initializer or not.
          relu: whether to use relu or not.
          stddev: standard deviation used for random weight initializer.
        Returns:
          A convolutional layer operation.
        """

        with tf.variable_scope(layer_name) as scope:
            channels = inputs.get_shape()[3]

            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer_conv2d()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            kernel = self._variable_with_weight_decay(
                'kernels', shape=[size, size, int(channels), filters], wd=0.0001, initializer=kernel_init,
                trainable=(not freeze))

            if w_bin == 8 and False:  # 8b quantization
                kernel_quant = self.lin_8b_quant(kernel)
                tf.summary.histogram('kernel_quant', kernel_quant)
                conv = tf.nn.conv2d(inputs, kernel_quant, [1, stride, stride, 1], padding=padding, name='convolution')

                if bias_on:
                    biases = self._variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
                    biases_quant = self.lin_8b_quant(biases)
                    tf.summary.histogram('biases_quant', biases_quant)
                    conv_bias = tf.nn.bias_add(conv, biases_quant, name='bias_add')
                else:
                    conv_bias = conv
            else:  # 16b quantization
                conv = tf.nn.conv2d(inputs, kernel, [1, stride, stride, 1], padding=padding, name='convolution')
                if bias_on:
                    biases = self._variable_on_device('biases', [filters], bias_init, trainable=(not freeze))
                    conv_bias = tf.nn.bias_add(conv, biases, name='bias_add')
                else:
                    conv_bias = conv

            if relu:
                out = tf.nn.relu(conv_bias, 'relu')
            else:
                out = conv_bias

            return out

    def _vgg_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True,
                   min_rng=-0.5, max_rng=0.5, bias_on=True, phase_train=True):
        with tf.variable_scope(layer_name):
            net = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=1, xavier=False,
                                   padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin,
                                   bias_on=bias_on)
            tf.summary.histogram('before_bn', net)
            net = self._batch_norm_tensor2('bn', net, phase_train=phase_train)  # BatchNorm
            tf.summary.histogram('before_relu', net)
            net = self.binary_wrapper(net, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # ReLU
            tf.summary.histogram('after_relu', net)
            if pool_en:
                pool = self._pooling_layer('pool', net, size=2, stride=2, padding='SAME')
            else:
                pool = net
            tf.summary.histogram('pool', pool)

            return pool

    @staticmethod
    def _variable_on_device(name, shape, initializer, trainable=True):
        """Helper to create a Variable.
        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        # TODO(bichen): fix the hard-coded data type below
        dtype = tf.float32
        if not callable(initializer):
            var = tf.get_variable(name, initializer=initializer, trainable=trainable)
        else:
            var = tf.get_variable(
                name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
        return var

    def _variable_with_weight_decay(self, name, shape, wd, initializer, trainable=True):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """
        var = self._variable_on_device(name, shape, initializer, trainable)

        return var

    def _fc_layer(self, layer_name, inputs, hiddens, flatten=False, relu=True, xavier=False, stddev=0.001, w_bin=16,
                  a_bin=16,
                  min_rng=0.0, max_rng=2.0):
        """Fully connected layer operation constructor.

        Args:
            layer_name: layer name.
            inputs: input tensor
            hiddens: number of (hidden) neurons in this layer.
            flatten: if true, reshape the input 4D tensor of shape
              (batch, height, weight, channel) into a 2D tensor with shape
              (batch, -1). This is used when the input to the fully connected layer
              is output of a convolutional layer.
            relu: whether to use relu or not.
            xavier: whether to use xavier weight initializer or not.
            stddev: standard deviation used for random weight initializer.
        Returns:
          A fully connected layer operation.
        """

        with tf.variable_scope(layer_name) as scope:
            input_shape = inputs.get_shape().as_list()
            if flatten:
                dim = input_shape[1] * input_shape[2] * input_shape[3]
                inputs = tf.reshape(inputs, [-1, dim])
            else:
                dim = input_shape[1]

            if xavier:
                kernel_init = tf.contrib.layers.xavier_initializer()
                bias_init = tf.constant_initializer(0.0)
            else:
                kernel_init = tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32)
                bias_init = tf.constant_initializer(0.0)

            weights = self._variable_with_weight_decay('weights', shape=[dim, hiddens], wd=0.0001,
                                                       initializer=kernel_init)
            biases = self._variable_on_device('biases', [hiddens], bias_init)

            if w_bin == 8:  # 8b quantization
                weights_quant = self.lin_8b_quant(weights)
            else:  # 16b quantization
                weights_quant = weights
            tf.summary.histogram('weights_quant', weights_quant)
            # ====================
            # no quantization on bias since it will be added to the 16b MUL output
            # ====================

            outputs = tf.nn.bias_add(tf.matmul(inputs, weights_quant), biases)
            tf.summary.histogram('outputs', outputs)

            if a_bin == 8:
                outputs_quant = self.lin_8b_quant(outputs, min_rng=min_rng, max_rng=max_rng)
            else:
                outputs_quant = outputs
            tf.summary.histogram('outputs_quant', outputs_quant)

            if relu:
                outputs = tf.nn.relu(outputs_quant, 'relu')
            return outputs

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0 or var.op.name.find(r'kernels') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    @staticmethod
    def _pooling_layer(
            layer_name, inputs, size, stride, padding='SAME'):
        """Pooling layer operation constructor.
        Args:
          layer_name: layer name.
          inputs: input tensor
          size: kernel size.
          stride: stride
          padding: 'SAME' or 'VALID'. See tensorflow doc for detailed description.
        Returns:
          A pooling layer operation.
        """

        with tf.variable_scope(layer_name) as scope:
            out = tf.nn.max_pool(inputs,
                                 ksize=[1, size, size, 1],
                                 strides=[1, stride, stride, 1],
                                 padding=padding)
            return out
