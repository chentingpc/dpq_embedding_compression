import os
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def get_arrays(data_dir):
    data = np.load(os.path.join(data_dir, "data-simplified.npz"))
    try:
        X_train = data["X_train"].astype(np.int32)
    except:  # For spitted large X_train, such as in Amazon_review_polarity
        X_train_part1 = data["X_train_part1"].astype(np.int32)
        X_train_part2 = data["X_train_part2"].astype(np.int32)
        X_train = np.concatenate((X_train_part1, X_train_part2))
    X_test = data["X_test"].astype(np.int32)
    y_train = data["y_train"].astype(np.int32)
    y_test = data["y_test"].astype(np.int32)
    vocab = data["vocab"]
    return X_train, y_train, X_test, y_test, vocab


def batchify_small(X, y, batch_size, num_epochs, reinitializer, is_train):
    """Cannot deal with large X."""
    y = tf.one_hot(y, len(set(y)))  # Transform to one-hot.
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_train:
        dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.batch(batch_size).repeat(num_epochs)

    if reinitializer:
        iterator = tf.compat.v1.data.Iterator.from_structure(
            dataset.output_types, dataset.output_shapes)
        initializer = iterator.make_initializer(dataset)
        X, y = iterator.get_next()
    else:
        X, y = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        initializer = None
    return X, y, initializer


def get_data_small(data_name, data_dir, batch_size):
    """Cannot deal with large X."""
    X_train, y_train, X_test, y_test, vocab = get_arrays(data_dir)
    X_train, y_train, _ = batchify_small(
        X_train, y_train, batch_size, None, False, True)
    X_test, y_test, reinitializer = batchify_small(
        X_test, y_test, batch_size, 1, True, False)
    return X_train, y_train, X_test, y_test, reinitializer, vocab


def batchify(X, y, batch_size, num_epochs, reinitializer, is_train):
    y = tf.one_hot(y, len(set(y)))  # Transform to one-hot.
    N = X.shape[0]
    dataset = tf.data.Dataset.range(N)
    if is_train:
        dataset = dataset.shuffle(buffer_size=N)
    dataset = dataset.batch(batch_size).repeat(num_epochs)

    if reinitializer:
        iterator = tf.compat.v1.data.Iterator.from_structure(
            tf.compat.v1.data.get_output_types(dataset), tf.compat.v1.data.get_output_shapes(dataset))
        initializer = iterator.make_initializer(dataset)
        idxs = iterator.get_next()
    else:
        idxs = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        initializer = None

    X, y = tf.nn.embedding_lookup(params=X, ids=idxs), tf.nn.embedding_lookup(params=y, ids=idxs)
    return X, y, initializer


def get_data(data_name, data_dir, batch_size):
    X_train_d, y_train, X_test_d, y_test, vocab = get_arrays(data_dir)
    num_classes = len(set(y_train))
    with tf.device("/cpu:0"):  # DEBUG
        X_train_holder = tf.compat.v1.placeholder(tf.int32, shape=X_train_d.shape)
        X_train = tf.Variable(X_train_holder, trainable=False)
        X_test_holder = tf.compat.v1.placeholder(tf.int32, shape=X_test_d.shape)
        X_test = tf.Variable(X_test_holder, trainable=False)
        X_train, y_train, _ = batchify(
            X_train, y_train, batch_size, None, False, True)
        X_test, y_test, reinitializer = batchify(
            X_test, y_test, batch_size, 1, True, False)
    return (X_train_d, X_test_d, X_train_holder, X_test_holder,
            X_train, y_train, X_test, y_test, reinitializer, vocab)


if __name__ == "__main__":
    import time
    home = os.path.expanduser("~")
    # data_name = "ag_news"
    data_name = "yahoo_answers"
    # data_name = "yelp_review_full"
    batch_size = 128
    data_dir = os.path.join(
        home, "corpus/text_classification_char_cnn/%s_csv" % data_name)
    (X_train_d, X_test_d, X_train_holder, X_test_holder, X_train, y_train,
        X_test, y_test, test_reinitializer, vocab) = get_data(
        data_name, data_dir, batch_size)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer(),
                 feed_dict={X_train_holder: X_train_d, X_test_holder: X_test_d})
        del X_train_d, X_test_d
        sess.run(test_reinitializer)
        start = time.time()
        for _ in range(1000):
            _ = sess.run([X_train, y_train])
        print("duration: ", time.time() - start)
        train_results = sess.run([X_train, y_train])
        test_results = sess.run([X_test, y_test])
        def get_info(sent): return " ".join([vocab[each] for each in sent])
    import pdb
    pdb.set_trace()
    print("ok")
