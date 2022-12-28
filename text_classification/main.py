import util
from model import Model
import data
import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TODO: avoid the OutOfRangeError msg.

sys.path.insert(0, "../lm")

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "")
flags.DEFINE_string("data_dir", None, "")
flags.DEFINE_string("save_dir", None, "")
flags.DEFINE_integer("max_iter", 10000, "")
flags.DEFINE_integer("eval_every_num_iter", 100, "")
flags.DEFINE_integer("batch_size", 1024, "")
flags.DEFINE_string("optimizer", "sgd", "")
flags.DEFINE_float("learning_rate", 1e-3, "")
flags.DEFINE_integer("dims", 100, "")
flags.DEFINE_bool("concat_maxpooling", False,
                  "Whether or not to use concat(mean_pooling, maxpooling).")
flags.DEFINE_integer("hidden_layers", 1, "Number of hidden layers.")
flags.DEFINE_float("reg_weight", 0, "")

# KDQ related
flags.DEFINE_string("kdq_type", "none", "one of none, vq, smx")
flags.DEFINE_integer("K", 16, "K-way code.")
flags.DEFINE_integer("D", 32, "D-dimensional code.")
flags.DEFINE_integer("kdq_d_in", 0, "adjustable query embedding size for smx")
flags.DEFINE_bool("kdq_share_subspace", False, "whether to share subspace")
flags.DEFINE_bool("additive_quantization", False, "only work with smx")


def main(_):
    (X_train_d, X_test_d, X_train_holder, X_test_holder, X_train, y_train,
        X_test, y_test, test_reinitializer, vocab) = data.get_data(
        FLAGS.dataset, FLAGS.data_dir, FLAGS.batch_size)
    flags.DEFINE_integer("vocab_size", len(vocab), 'Auto add vocab size')

    with tf.compat.v1.name_scope("Train"):
        with tf.compat.v1.variable_scope("model", reuse=False):
            m = Model()
            loss_train, preds_train, train_op = m.forward(
                X_train, y_train, is_training=True)
    with tf.compat.v1.name_scope("Test"):
        with tf.compat.v1.variable_scope("model", reuse=True):
            loss_test, preds_test, _ = m.forward(
                X_test, y_test, is_training=False)

    # Verbose.
    print("FLAGS:")
    for key, value in tf.compat.v1.flags.FLAGS.__flags.items():
        print(key, value._value)
    print("Number of trainable params: {}".format(util.get_parameter_count()))
    print(tf.compat.v1.trainable_variables())

    # Training session.
    init_feed_dict = {X_train_holder: X_train_d, X_test_holder: X_test_d}
    sv = tf.compat.v1.train.Supervisor(saver=None, init_feed_dict=init_feed_dict)
    config_proto = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    with sv.managed_session(config=config_proto) as sess:
        del X_train_d, X_test_d
        print("Start training")
        # Training loop.
        losses_train_ = []
        test_accs = []
        start = time.time()
        for it in range(1 + FLAGS.max_iter):
            loss_train_, _ = sess.run([loss_train, train_op])
            losses_train_.append(loss_train_)

            # Evaluation.
            if it % FLAGS.eval_every_num_iter == 0:
                sess.run(test_reinitializer)
                train_hits = []
                test_hits = []
                for _ in range(100):
                    results = sess.run([y_train, preds_train])
                    train_hits.append(np.argmax(results[0], 1) == results[1])
                train_accuracy = np.concatenate(train_hits).mean()
                while True:
                    try:
                        results = sess.run([y_test, preds_test])
                        test_hits.append(np.argmax(results[0], 1) == results[1])
                    except tf.errors.OutOfRangeError:
                        break
                test_accuracy = np.concatenate(test_hits).mean()
                test_accs.append(test_accuracy)
                end = time.time()
                print("Iter {:6}, {:.3} (secs), loss {:.4}, train acc {:.4} test acc {:.4}".format(
                    it, end - start, np.mean(losses_train_), train_accuracy, test_accuracy))
                losses_train_ = []
                start = time.time()
        print("Best test accuracy {}".format(max(test_accs)))


if __name__ == "__main__":
    tf.compat.v1.app.run()
