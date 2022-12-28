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
"""Utilities for Grappler autoparallel optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial, reduce
import tensorflow as tf
import tensorflow_addons as tfa

FLAGS = tf.compat.v1.flags.FLAGS

eps_micro = 1e-15  # tf.float32 sensible.
eps_tiny = 1e-10   # tf.float32 sensible.
eps_small = 3e-8   # tf.float16 sensible.


def get_activation(name):
    """Returns activation function given name."""
    name = name.lower()
    if name == "relu":
        return tf.nn.relu
    elif name == "sigmoid":
        return tf.nn.sigmoid
    elif name == "tanh":
        return tf.nn.sigmoid
    elif name == "elu":
        return tf.nn.elu
    elif name == "linear":
        return lambda x: x
    else:
        raise ValueError("Unknown activation name {}".format(name))

    return name


def get_optimizer(name):
    name = name.lower()
    if name == "sgd":
        optimizer = tf.compat.v1.train.GradientDescentOptimizer
    elif name == "momentum":
        optimizer = partial(tf.compat.v1.train.MomentumOptimizer,
                            momentum=0.05, use_nesterov=True)
    elif name == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer
        # optimizer = partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9)
    elif name == "lazy_adam":
        optimizer = tfa.optimizers.LazyAdam # TODO: migrate to tf2, global_step not supported
        # optimizer = partial(tf.contrib.opt.LazyAdamOptimizer, beta1=0.5, beta2=0.9)
    elif name == "adagrad":
        optimizer = tf.compat.v1.train.AdagradOptimizer
    elif name == "rmsprop":
        optimizer = tf.compat.v1.train.RMSPropOptimizer
    else:
        raise ValueError("Unknown optimizer name {}.".format(name))

    return optimizer


def get_parameter_count(excludings=None, display_count=True):
    trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    count = 0
    for var in trainables:
        ignored = False
        if excludings is not None:
            for excluding in excludings:
                if var.name.find(excluding) >= 0:
                    ignored = True
                    break
        if ignored:
            continue
        if var.shape == tf.TensorShape(None):
            tf.compat.v1.logging.warn("var {} has unknown shape and it is not counted.".format(
                var.name))
            continue
        if var.shape.as_list() == []:
            count_ = 1
        else:
            count_ = reduce(lambda x, y: x * y, var.shape.as_list())
        count += count_
        if display_count:
            print("{0:80} {1}".format(
                var.name, count_))
    return count
