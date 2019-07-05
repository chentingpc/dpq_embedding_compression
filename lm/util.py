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
from functools import partial
import tensorflow as tf

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.flags.FLAGS

eps_micro = 1e-15  # tf.float32 sensible.
eps_tiny = 1e-10   # tf.float32 sensible.
eps_small = 3e-8   # tf.float16 sensible.


def export_state_tuples(state_tuples, name):
  for state_tuple in state_tuples:
    tf.add_to_collection(name, state_tuple.c)
    tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
  restored = []
  for i in range(len(state_tuples) * num_replicas):
    c = tf.get_collection_ref(name)[2 * i + 0]
    h = tf.get_collection_ref(name)[2 * i + 1]
    restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
  return tuple(restored)


def with_prefix(prefix, name):
  """Adds prefix to name."""
  return "/".join((prefix, name))


def with_autoparallel_prefix(replica_id, name):
  return with_prefix("AutoParallel-Replica-%d" % replica_id, name)


class UpdateCollection(object):
  """Update collection info in MetaGraphDef for AutoParallel optimizer."""

  def __init__(self, metagraph, model):
    self._metagraph = metagraph
    self.replicate_states(model.initial_state_name)
    self.replicate_states(model.final_state_name)
    self.update_snapshot_name("variables")
    self.update_snapshot_name("trainable_variables")

  def update_snapshot_name(self, var_coll_name):
    var_list = self._metagraph.collection_def[var_coll_name]
    for i, value in enumerate(var_list.bytes_list.value):
      var_def = variable_pb2.VariableDef()
      var_def.ParseFromString(value)
      # Somehow node Model/global_step/read doesn't have any fanout and seems to
      # be only used for snapshot; this is different from all other variables.
      if var_def.snapshot_name != "Model/global_step/read:0":
        var_def.snapshot_name = with_autoparallel_prefix(
            0, var_def.snapshot_name)
      value = var_def.SerializeToString()
      var_list.bytes_list.value[i] = value

  def replicate_states(self, state_coll_name):
    state_list = self._metagraph.collection_def[state_coll_name]
    num_states = len(state_list.node_list.value)
    for replica_id in range(1, FLAGS.num_gpus):
      for i in range(num_states):
        state_list.node_list.value.append(state_list.node_list.value[i])
    for replica_id in range(FLAGS.num_gpus):
      for i in range(num_states):
        index = replica_id * num_states + i
        state_list.node_list.value[index] = with_autoparallel_prefix(
            replica_id, state_list.node_list.value[index])


def auto_parallel(metagraph, model):
  from tensorflow.python.grappler import tf_optimizer
  rewriter_config = rewriter_config_pb2.RewriterConfig()
  rewriter_config.optimizers.append("autoparallel")
  rewriter_config.auto_parallel.enable = True
  rewriter_config.auto_parallel.num_replicas = FLAGS.num_gpus
  optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
  metagraph.graph_def.CopyFrom(optimized_graph)
  UpdateCollection(metagraph, model)


def safer_log(x, eps=eps_micro):
  """Avoid nan when x is zero by adding small eps.
  
  Note that if x.dtype=tf.float16, \forall eps, eps < 3e-8, is equal to zero.
  """
  return tf.log(x + eps)


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


def hparam_fn(hparams, prefix=None):
  """Returns a function to get hparam with prefix in the name."""
  if prefix is None or prefix == "":
    prefix = ""
  elif isinstance(prefix, str):
    prefix += "_"
  else:
    raise ValueError("prefix {} is invalid".format(prefix))
  get_hparam = lambda name: getattr(hparams, prefix + name)
  return get_hparam


def filter_activation_fn():
  """Returns a activation if actv is an string name, otherwise input."""
  filter_actv = lambda actv: (
      get_activation(actv) if isinstance(actv, str) else actv)
  return filter_actv


def get_optimizer(name):
  name = name.lower()
  if name == "sgd":
    optimizer = tf.train.GradientDescentOptimizer
  elif name == "momentum":
    optimizer = partial(tf.train.MomentumOptimizer,
                        momentum=0.05, use_nesterov=True)
  elif name == "adam":
    optimizer = tf.train.AdamOptimizer
    # optimizer = partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9)
  elif name == "lazy_adam":
    optimizer = tf.contrib.opt.LazyAdamOptimizer
    # optimizer = partial(tf.contrib.opt.LazyAdamOptimizer, beta1=0.5, beta2=0.9)
  elif name == "adagrad":
    optimizer = tf.train.AdagradOptimizer
  elif name == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer
  else:
    raise ValueError("Unknown optimizer name {}.".format(name))

  return optimizer


def replace_list_element(data_list, x, y):
  return [y if each == x else each for each in data_list]


def get_parameter_count(excludings=None, display_count=True):
  trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
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
      tf.logging.warn("var {} has unknown shape and it is not counted.".format(
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


def save_emb_visualize_meta(save_path,
                            emb_var_names,
                            label_lists,
                            metadata_names=None):
  """Save meta information about the embedding visualization in tensorboard.

  Args:
    save_path: a `string` specifying the save location.
    emb_var_names: a `list` containing variable names.
    label_lists: a `list` of lists of labels, each label list corresponds to an
      emb_var_name.
    metadata_names: a `list` of file names for metadata, if not specify, will
      use emb_var_names.
  """
  if not isinstance(emb_var_names, list):
    raise ValueError("emb_var_names must be a list of var names.")
  if not isinstance(label_lists, list) or not isinstance(label_lists[0], list):
    raise ValueError("label_lists must be a list of label lists.")

  if metadata_names is None:
    metadata_names = emb_var_names

  config = projector.ProjectorConfig()
  for emb_var_name, metadata_name, labels in zip(
      emb_var_names, metadata_names, label_lists):
    filename = "metadata-{}.tsv".format(metadata_name)
    metadata_path = os.path.join(save_path, filename)
    with open(metadata_path, "w") as fp:
      for label in labels:
        fp.write("{}\n".format(label))
    embedding = config.embeddings.add()
    embedding.tensor_name = emb_var_name
    embedding.metadata_path = metadata_path
  summary_writer = tf.summary.FileWriter(save_path)
  projector.visualize_embeddings(summary_writer, config)


def create_labels_based_on_codes(codes, K):
  """Take codes and produce per-axis based and prefix based labels.

  Args:
    codes: a `np.ndarray` of size (N, D) where N data points with D-dimensional
      discrete code.
    K: a `int` specifying the cardinality of the code in each dimension.

  Returns:
    label_lists: a list of labels.
  """
  N, D = codes.shape
  label_lists = []

  # Create per-axis labels.
  for i in range(D):
    label_lists.append(codes[:, i].tolist())

  # Create prefix labels.
  buffer_basis = 0
  for i in range(D):
    buffer_basis = buffer_basis * K + codes[:, i]
    label_lists.append(buffer_basis.tolist())

  return label_lists
