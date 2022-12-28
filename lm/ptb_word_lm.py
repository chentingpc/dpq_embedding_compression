# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Modified based on https://github.com/tensorflow/models/tree/r1.4.0/tutorials/rnn/ptb.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import inspect

from collections import Counter, defaultdict
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import reader
import util

parent_path = "/".join(os.getcwd().split('/')[:-1])
sys.path.append(os.path.join(parent_path, "core"))
from kd_quantizer import KDQuantizer
from kdq_embedding import full_embed, kdq_embed, KDQhparam


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable when using tf.Print

flags = tf.compat.v1.flags
logging = tf.compat.v1.logging

# Define basics.
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("dataset", "ptb",
                    "Supported dataset: ptb, text8, 1billion.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_integer("save_model_secs", 0,
                     "Setting it zero to avoid saving the checkpoint.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats.")
flags.DEFINE_integer("max_max_epoch", None,
                     "The total number of epochs for training.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
flags.DEFINE_float("max_grad_norm", None,
                   "the maximum permissible norm of the gradient.")
flags.DEFINE_float("lr_decay", None, "lr decay reate.")
flags.DEFINE_integer("eval_topk", 10, "topk recall/precision to compute.")

# KDQ related
flags.DEFINE_string("kdq_type", "none", "one of none, vq, smx")
flags.DEFINE_integer("K", 16, "K-way code.")
flags.DEFINE_integer("D", 32, "D-dimensional code.")
flags.DEFINE_integer("kdq_d_in", 0, "adjustable query embedding size for smx")
flags.DEFINE_bool("kdq_share_subspace", False, "whether to share subspace")
flags.DEFINE_bool("additive_quantization", False, "only work with smx")

FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

TRAIN_MODE = "train"
VALID_MODE = "valid"
TEST_MODE = "test"

vocab_size = None


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


def print_at_beginning(hparams):
    global vocab_size
    print("kdq_type={}, vocab_size={}, K={}, D={}".format(
        FLAGS.kdq_type, vocab_size, FLAGS.K, FLAGS.D))
    print("Number of trainable params:    {}".format(
        util.get_parameter_count(
            excludings=["code_logits", "embb", "symbol2code"])))


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_, vocab_size):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        self._config = config
        size = config.hidden_size

        if FLAGS.kdq_type == "none":
            inputs = full_embed(input_.input_data, vocab_size, size)
        else:
            kdq_hparam = KDQhparam(
                K=FLAGS.K, D=FLAGS.D, kdq_type=FLAGS.kdq_type,
                kdq_d_in=FLAGS.kdq_d_in, kdq_share_subspace=FLAGS.kdq_share_subspace,
                additive_quantization=FLAGS.additive_quantization)
            inputs = kdq_embed(
                input_.input_data, vocab_size, size, kdq_hparam, is_training)

        # RNN layers
        outputs, state = self._build_rnn_graph(inputs, config, is_training)

        # final softmax layer
        targets = input_.targets
        softmax_w = tf.compat.v1.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.compat.v1.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.compat.v1.nn.xw_plus_b(outputs, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits,
                            [self.batch_size, self.num_steps, vocab_size])
        loss = tfa.seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=False)  # (batch_size, num_steps)

        # Update the cost
        self._nll = tf.reduce_sum(input_tensor=tf.reduce_mean(input_tensor=loss, axis=0))
        self._cost = self._nll
        self._final_state = state

        # compute recall metric
        _, preds_topk = tf.nn.top_k(logits, FLAGS.eval_topk)
        targets_topk = tf.tile(
            tf.expand_dims(targets, -1),
            [1] * targets.shape.ndims + [FLAGS.eval_topk])
        hits = tf.reduce_sum(
            input_tensor=tf.cast(tf.equal(preds_topk, targets_topk), tf.float32), axis=-1)
        self._recall_at_k = tf.reduce_sum(input_tensor=tf.reduce_mean(input_tensor=hits, axis=0))

        if not is_training:
            return

        # Add regularization.
        print("[INFO] regularization loss",
              tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
        self._cost += sum(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

        # Optimizer
        self._lr = tf.Variable(0.0, trainable=False)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = tf.gradients(ys=self._cost, xs=tf.compat.v1.trainable_variables())
            tf.compat.v1.summary.scalar("global_grad_norm", tf.linalg.global_norm(grads))
            grads, _ = tf.clip_by_global_norm(grads,
                                              config.max_grad_norm)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tf.compat.v1.trainable_variables()),)
                # global_step=tf.compat.v1.train.get_or_create_global_step())
        self._new_lr = tf.compat.v1.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.compat.v1.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, rate=1 - (config.keep_prob))

        inputs = tf.transpose(a=inputs, perm=[1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.compat.v1.get_variable(
            "lstm_params",
            initializer=tf.random.uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        self._initial_state = (tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(a=outputs, perm=[1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.nn.rnn_cell.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
                config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
            return tfa.rnn.LayerNormLSTMCell(config.hidden_size)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells without Wrapper."""
        init_sates = []
        final_states = []
        with tf.compat.v1.variable_scope("RNN", reuse=not is_training):
            for l in range(config.num_layers):
                with tf.compat.v1.variable_scope("layer_%d" % l):
                    cell = self._get_lstm_cell(config, is_training)
                    initial_state = cell.get_initial_state(batch_size=self.batch_size, dtype=data_type())
                    init_sates.append(initial_state)
                    state = init_sates[-1]
                    if is_training and config.keep_prob < 1:
                        inputs = tf.nn.dropout(inputs, rate=1 - (config.keep_prob))
                    inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
                    outputs, state = tf.compat.v1.nn.static_rnn(cell, inputs,
                                                               initial_state=init_sates[-1])
                    final_states.append(state)
                    outputs = [tf.expand_dims(output, 1) for output in outputs]
                    outputs = tf.concat(outputs, 1)
                    inputs = outputs
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        self._initial_state = tuple(init_sates)
        state = tuple(final_states)
        if is_training and config.keep_prob < 1:
            outputs = tf.nn.dropout(outputs, rate=1 - (config.keep_prob))
        return outputs, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def nll(self):
        return self._nll

    @property
    def recall_at_k(self):
        return self._recall_at_k

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 100 if FLAGS.max_max_epoch is None else FLAGS.max_max_epoch
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    global vocab_size
    rnn_mode = BLOCK


class MediumConfig(SmallConfig):
    """Medium config."""
    init_scale = 0.05
    max_grad_norm = 5 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39 if FLAGS.max_max_epoch is None else FLAGS.max_max_epoch
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    global vocab_size
    rnn_mode = BLOCK


class LargeConfig(SmallConfig):
    """Large config."""
    init_scale = 0.04
    max_grad_norm = 10 if FLAGS.max_grad_norm is None else FLAGS.max_grad_norm
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55 if FLAGS.max_max_epoch is None else FLAGS.max_max_epoch
    keep_prob = 0.35
    lr_decay = 1 / 1.15 if (FLAGS.lr_decay is None or FLAGS.lr_decay < 0) else (
        FLAGS.lr_decay)
    batch_size = 20
    global vocab_size
    rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False, mode=TRAIN_MODE):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    nlls = 0.0
    recalls_at_k = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "nll": model.nll,
        "recall_at_k": model.recall_at_k,
        "final_state": model.final_state
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i][0]
            feed_dict[h] = state[i][1]

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        nll = vals["nll"]
        recall_at_k = vals["recall_at_k"]
        state = vals["final_state"]

        costs += cost
        nlls += nll
        recalls_at_k += recall_at_k
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f cost %.3f perplexity: %.3f recall@%d: %.3f speed: %.0f wps" %
                  (step*1./model.input.epoch_size, costs / iters, np.exp(nlls/iters),
                   FLAGS.eval_topk, recalls_at_k / iters,
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(nlls / iters), recalls_at_k / iters


def get_config(verbose=False):
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "medium":
        config = MediumConfig()
    elif FLAGS.model == "large":
        config = LargeConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if verbose:
        config_attrs = [a for a in inspect.getmembers(config) if not (
            a[0].startswith('__') and a[0].endswith('__'))]
        print(config_attrs)
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    raw_data = reader.ptb_raw_data(FLAGS.dataset, FLAGS.data_path, True)
    train_data, valid_data, test_data, _vocab_size, _id2word = raw_data
    global vocab_size
    vocab_size = _vocab_size

    config = get_config(verbose=True)
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.compat.v1.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.compat.v1.name_scope("Train"):
            train_input = PTBInput(config=config, data=train_data, name="TrainInput")
            with tf.compat.v1.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True,
                             config=config,
                             input_=train_input,
                             vocab_size=vocab_size)
            tf.compat.v1.summary.scalar("Training Loss", m.cost)
            tf.compat.v1.summary.scalar("Learning Rate", m.lr)

        with tf.compat.v1.name_scope("Valid"):
            valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
            with tf.compat.v1.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False,
                                  config=config,
                                  input_=valid_input,
                                  vocab_size=vocab_size)
            tf.compat.v1.summary.scalar("Validation Loss", mvalid.cost)

        with tf.compat.v1.name_scope("Test"):
            test_input = PTBInput(
                config=eval_config, data=test_data, name="TestInput")
            with tf.compat.v1.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False,
                                 config=eval_config,
                                 input_=test_input,
                                 vocab_size=vocab_size)

        models = {"Train": m, "Valid": mvalid, "Test": mtest}

        print_at_beginning(config)
        sv = tf.compat.v1.train.Supervisor(logdir=FLAGS.save_path,
                                 save_model_secs=FLAGS.save_model_secs,
                                 save_summaries_secs=10)
        config_proto = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config_proto.gpu_options.allow_growth = True
        with sv.managed_session(config=config_proto) as session:
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                train_perplexity, train_recall_at_k = run_epoch(
                    session, m, eval_op=m.train_op, verbose=True, mode=TRAIN_MODE)
                print("Epoch: %d Train Perplexity: %.3f, recall@%d: %.3f" % (
                    i + 1, train_perplexity, FLAGS.eval_topk, train_recall_at_k))
                valid_perplexity, valid_recall_at_k = run_epoch(
                    session, mvalid, mode=VALID_MODE)
                print("Epoch: %d Valid Perplexity: %.3f, recall@%d: %.3f" % (
                    i + 1, valid_perplexity, FLAGS.eval_topk, valid_recall_at_k))

            test_perplexity, test_recall_at_k = run_epoch(
                session, mtest, mode=TEST_MODE)
            print("Test Perplexity: %.3f, recall@%d: %.3f" % (
                test_perplexity, FLAGS.eval_topk, test_recall_at_k))

            if FLAGS.save_path and sv.saver is not None:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session,
                              os.path.join(FLAGS.save_path, "model"),
                              global_step=sv.global_step)


if __name__ == "__main__":
    tf.compat.v1.app.run()
