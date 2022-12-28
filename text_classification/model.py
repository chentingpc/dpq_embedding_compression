import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import util
parent_path = "/".join(os.getcwd().split('/')[:-1])
sys.path.append(os.path.join(parent_path, "core"))
from kd_quantizer import KDQuantizer
from kdq_embedding import full_embed, kdq_embed, KDQhparam

FLAGS = tf.compat.v1.flags.FLAGS


class Model(object):
    def __init__(self):
        pass

    def forward(self, features, labels, is_training=False):
        """Returns loss, preds, train_op.

        Args:
          features: (batch_size, max_seq_length)
          labels: (batch_size, num_classes)

        Returns:
          loss: (batch_size, )
          preds: (batch_size, )
          train_op: op.
        """
        num_classes = labels.shape.as_list()[-1]
        batch_size = tf.shape(input=features)[0]
        mask = tf.cast(tf.greater(features, 0), tf.float32)  # (bs, max_seq_length)
        lengths = tf.reduce_sum(input_tensor=mask, axis=1, keepdims=True)  # (batch_size, 1)

        # Embedding
        if FLAGS.kdq_type == "none":
            inputs = full_embed(features, FLAGS.vocab_size, FLAGS.dims)
        else:
            kdq_hparam = KDQhparam(
                K=FLAGS.K, D=FLAGS.D, kdq_type=FLAGS.kdq_type,
                kdq_d_in=FLAGS.kdq_d_in, kdq_share_subspace=FLAGS.kdq_share_subspace,
                additive_quantization=FLAGS.additive_quantization)
            inputs = kdq_embed(
                features, FLAGS.vocab_size, FLAGS.dims, kdq_hparam, is_training)
        word_embs = inputs  # (bs, length, emb_dim)
        word_embs *= tf.expand_dims(mask, -1)

        embs_maxpool = tf.reduce_max(input_tensor=word_embs, axis=1)  # Max pooling.
        embs_meanpool = tf.reduce_sum(input_tensor=word_embs, axis=1) / lengths  # Mean pooling.
        if FLAGS.concat_maxpooling:
            embs = tf.concat([embs_meanpool, embs_maxpool], -1)
        else:
            embs = embs_meanpool
        if FLAGS.hidden_layers > 0:
            embs = tf.nn.relu(
                tf.compat.v1.layers.batch_normalization(embs, training=is_training))
            embs = tf.compat.v1.layers.dense(embs, FLAGS.dims)
            embs = tf.nn.relu(
                tf.compat.v1.layers.batch_normalization(embs, training=is_training))
        logits = tf.compat.v1.layers.dense(embs, num_classes)
        preds = tf.argmax(input=logits, axis=-1)[:batch_size]
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        if is_training:
            # Regular loss updater.
            loss_scalar = tf.reduce_mean(input_tensor=loss)
            loss_scalar += FLAGS.reg_weight * tf.reduce_mean(input_tensor=word_embs**2)
            loss_scalar += tf.reduce_sum(
                input_tensor=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))
            op = util.get_optimizer(FLAGS.optimizer)(learning_rate=FLAGS.learning_rate)
            train_op = op.minimize(
                loss=loss_scalar,
                global_step=tf.compat.v1.train.get_or_create_global_step(),
                var_list=tf.compat.v1.trainable_variables())
        else:
            train_op = False
            loss_scalar = None

        return loss_scalar, preds, train_op
