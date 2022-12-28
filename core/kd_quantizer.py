import tensorflow as tf


def safer_log(x, eps=1e-10):
    """Avoid nan when x is zero by adding small eps.

    Note that if x.dtype=tf.float16, \forall eps, eps < 3e-8, is equal to zero.
    """
    return tf.compat.v1.log(x + eps)


def sample_gumbel(shape):
    """Sample from Gumbel(0, 1)"""
    U = tf.compat.v1.random_uniform(shape, minval=0, maxval=1)
    return -safer_log(-safer_log(U))


class KDQuantizer(object):
    def __init__(self, K, D, d_in, d_out, tie_in_n_out,
                 query_metric="dot", shared_centroids=False,
                 beta=0., tau=1.0, softmax_BN=True):
        """
        Args:
          K, D: int, size of KD code.
          d_in: dim of continuous input for each of D axis.
          d_out: dim of continuous output for each of D axis.
          tie_in_n_out: boolean, whether or not to tie the input/output centroids.
            If True, it is vector quantization, else it is tempering softmax.
          query_metric: string, which metric to use for input centroid matching.
          shared_centroids: boolean, whether or not to share centroids for
            different bits in D.
          beta: float, KDQ regularization coefficient.
          tau: float or None, (tempering) softmax temperature.
            If None, set to learnable.
          softmax_BN: whether to use BN in (tempering) softmax.
        """
        self._K = K
        self._D = D
        self._d_in = d_in
        self._d_out = d_out
        self._tie_in_n_out = tie_in_n_out
        self._query_metric = query_metric
        self._shared_centroids = shared_centroids
        self._beta = beta
        if tau is None:
            self._tau = tf.get_variable(
                "tau", [], initializer=tf.constant_initializer(1.0))
        else:
            self._tau = tf.constant(tau)
        self._softmax_BN = softmax_BN

        # Create centroids for keys and values.
        D_to_create = 1 if shared_centroids else D
        centroids_k = tf.compat.v1.get_variable(
            "centroids_k", [D_to_create, K, d_in])
        if tie_in_n_out:
            centroids_v = centroids_k
        else:
            centroids_v = tf.compat.v1.get_variable(
                "centroids_v", [D_to_create, K, d_out])
        if shared_centroids:
            centroids_k = tf.tile(centroids_k, [D, 1, 1])
            if tie_in_n_out:
                centroids_v = centroids_k
            else:
                centroids_v = tf.tile(centroids_v, [D, 1, 1])
        self._centroids_k = centroids_k
        self._centroids_v = centroids_v

    def forward(self,
                inputs,
                sampling=False,
                is_training=True):
        """Returns quantized embeddings from centroids.

        Args:
          inputs: embedding tensor of shape (batch_size, D, d_in)

        Returns:
          code: (batch_size, D)
          embs_quantized: (batch_size, D, d_out)
        """
        with tf.name_scope("kdq_forward"):
            # House keeping.
            centroids_k = self._centroids_k  # (D, K, d_in)
            centroids_v = self._centroids_v

            # Compute distance (in a metric) between inputs and centroids_k
            # the response is in the shape of (batch_size, D, K)
            if self._query_metric == "euclidean":
                norm_1 = tf.reduce_sum(inputs**2, -1, keep_dims=True)  # (bs, D, 1)
                norm_2 = tf.expand_dims(tf.reduce_sum(centroids_k**2, -1), 0)  # (1, D, K)
                dot = tf.matmul(tf.transpose(inputs, perm=[1, 0, 2]),
                                tf.transpose(centroids_k, perm=[0, 2, 1]))  # (D, bs, K)
                response = -norm_1 + 2 * tf.transpose(dot, perm=[1, 0, 2]) - norm_2
            elif self._query_metric == "cosine":
                inputs = tf.nn.l2_normalize(inputs, -1)
                centroids_k = tf.nn.l2_normalize(centroids_k, -1)
                response = tf.matmul(tf.transpose(inputs, perm=[1, 0, 2]),
                                     tf.transpose(centroids_k, perm=[0, 2, 1]))  # (D, bs, K)
                response = tf.transpose(response, perm=[1, 0, 2])
            elif self._query_metric == "dot":
                response = tf.matmul(tf.transpose(inputs, perm=[1, 0, 2]),
                                     tf.transpose(centroids_k, perm=[0, 2, 1]))  # (D, bs, K)
                response = tf.transpose(response, perm=[1, 0, 2])
            else:
                raise ValueError("Unknown metric {}".format(self._query_metric))
            response = tf.reshape(response, [-1, self._D, self._K])
            if self._softmax_BN:
                # response = tf.contrib.layers.instance_norm(
                #    response, scale=False, center=False,
                #    trainable=False, data_format="NCHW")
                response = tf.compat.v1.layers.batch_normalization(
                    response, scale=False, center=False, training=is_training)
                # Layer norm as alternative to BN.
                # response = tf.contrib.layers.layer_norm(
                #    response, scale=False, center=False)
            response_prob = tf.nn.softmax(response / self._tau, -1)

            # Compute the codes based on response.
            codes = tf.argmax(response, -1)  # (batch_size, D)
            if sampling:
                response = safer_log(response_prob)
                noises = sample_gumbel(tf.shape(response))
                neighbor_idxs = tf.argmax(response + noises, -1)  # (batch_size, D)
            else:
                neighbor_idxs = codes

            # Compute the outputs, which has shape (batch_size, D, d_out)
            if self._tie_in_n_out:
                if not self._shared_centroids:
                    D_base = tf.convert_to_tensor(
                        [self._K*d for d in range(self._D)], dtype=tf.int64)
                    neighbor_idxs += tf.expand_dims(D_base, 0)  # (batch_size, D)
                neighbor_idxs = tf.reshape(neighbor_idxs, [-1])  # (batch_size * D)
                centroids_v = tf.reshape(centroids_v, [-1, self._d_out])
                outputs = tf.nn.embedding_lookup(centroids_v, neighbor_idxs)
                outputs = tf.reshape(outputs, [-1, self._D, self._d_out])
                outputs_final = tf.stop_gradient(outputs - inputs) + inputs
            else:
                nb_idxs_onehot = tf.one_hot(neighbor_idxs,
                                            self._K)  # (batch_size, D, K)
                nb_idxs_onehot = response_prob - tf.stop_gradient(
                    response_prob - nb_idxs_onehot)
                # nb_idxs_onehot = response_prob  # use continuous output
                outputs = tf.matmul(
                    tf.transpose(nb_idxs_onehot, [1, 0, 2]),  # (D, bs, K)
                    centroids_v)  # (D, bs, d)
                outputs_final = tf.transpose(outputs, [1, 0, 2])

            # Add regularization for updating centroids / stabilization.
            if is_training:
                print("[INFO] Adding KDQ regularization.")
                if self._tie_in_n_out:
                    alpha = 1.
                    beta = self._beta
                    gamma = 0.0
                    reg = alpha * tf.reduce_mean(
                        (outputs - tf.stop_gradient(inputs))**2, name="centroids_adjust")
                    reg += beta * tf.reduce_mean(
                        (tf.stop_gradient(outputs) - inputs)**2, name="input_commit")
                    minaxis = [0, 1] if self._shared_centroids else [0]
                    reg += gamma * tf.reduce_mean(  # could sg(inputs), but still not eff.
                        tf.reduce_min(-response, minaxis), name="de_isolation")
                else:
                    beta = self._beta
                    reg = - beta * tf.reduce_mean(
                        tf.reduce_sum(nb_idxs_onehot * safer_log(response_prob), [2]))
                    # entropy regularization
                    # reg = - beta * tf.reduce_mean(
                    #    tf.reduce_sum(response_prob * safer_log(response_prob), [2]))
                tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, reg)

            return codes, outputs_final


if __name__ == "__main__":
    # VQ
    with tf.compat.v1.variable_scope("VQ"):
        kdq_demo = KDQuantizer(100, 10, 5, 5, True, "euclidean")
    codes_vq, outputs_vq = kdq_demo.forward(tf.random_normal([64, 10, 5]))
    # tempering softmax
    with tf.compat.v1.variable_scope("tempering_softmax"):
        kdq_demo = KDQuantizer(100, 10, 5, 10, False, "dot")
    codes_ts, outputs_ts = kdq_demo.forward(tf.random_normal([64, 10, 5]))
