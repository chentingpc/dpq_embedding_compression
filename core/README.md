Core modules for KDQ
================================================================================

This folder contains core modules for KDQ.

## KDQuantizer explained

This supports both VQ (in-space quantization), and SMX (out-of-space quantization), which quantize  continuous space in terms of KD codes.

A typical full embedding is a function as follows.

```python
def full_embedding(input, vocab_size, emb_size, hparams=None,
                   training=True, name="full_emb"):
  """Full embedding baseline.

  Args:
    input: int multi-dim tensor, entity idxs.
    vocab_size: int, vocab size
    emb_size: int, output embedding size

  Returns:
    input_emb: float tensor, embedding for entity idxs.
  """
  with tf.variable_scope(name):
    embedding = tf.get_variable("embedding", [vocab_size, emb_size])
    input_emb = tf.nn.embedding_lookup(embedding, input)
  return input_emb
```

An drop-in replacement of the full embedding with KDQ embedding is given as follows.

```python
def kdq_embedding(input, vocab_size, emb_size, hparams=None,
                  training=True, name="kdq_emb"):
  """KDQ embedding with VQ or SMX.

  This is an drop-in replacement of ``full_embedding`` baseline above.

  Args:
    input: int multi-dim tensor, entity idxs.
    vocab_size: int, vocab size
    emb_size: int, output embedding size
    hparams: hparams for KDQ, see KDQhparam class for a reference.
    training: whether or not this is in training mode (related to BN)

  Returns:
    input_emb: float tensor, embedding for entity idxs.
  """
  if hparams is None:
    hparams = KDQhparam()
  d, K, D = emb_size, hparams.K, hparams.D
  d_in = d//D if hparams.kdq_d_in <= 0 else hparams.kdq_d_in  # could use diff. d_in/d_out for smx
  d_out = d if hparams.additive_quantization else d//D
  out_size = [D, emb_size] if hparams.additive_quantization else [emb_size]

  with tf.variable_scope(name):
    query_wemb = tf.get_variable(
        "query_wemb", [vocab_size, D * d_in], dtype=tf.float32)
    idxs = tf.reshape(input, [-1])
    input_emb = tf.nn.embedding_lookup(query_wemb, idxs)  # (bs*len, d)

    if hparams.kdq_type == "vq":
        assert hparams.kdq_d_in <= 0, (
            "kdq d_in cannot be changed (to %d) for vq" % hparams.kdq_d_in)
        tie_in_n_out = True
        dist_metric = "euclidean"
        beta, tau, softmax_BN = 0.0, 1.0, True
        share_subspace = hparams.kdq_share_subspace
    else:
        assert hparams.kdq_type == "smx", [
            "unknown kdq_type %s" % hparams.kdq_type]
        tie_in_n_out = False
        dist_metric = "dot"
        beta, tau, softmax_BN = 0.0, 1.0, True
        share_subspace = hparams.kdq_share_subspace
        kdq = KDQuantizer(K, D, d_in, d_out, tie_in_n_out,
                          dist_metric, share_subspace,
                          beta, tau, softmax_BN)
    _, input_emb = kdq.forward(tf.reshape(input_emb, [-1, D, d_in]),
                               is_training=training)
    final_size = tf.concat(
        [tf.shape(input), tf.constant(out_size)], 0)
    input_emb = tf.reshape(input_emb, final_size)
    if hparams.additive_quantization:
      input_emb = tf.reduce_mean(input_emb, -2)
  return input_emb


class KDQhparam(object):
  # A default KDQ parameter setting (demo)
  K = 16
  D = 32
  kdq_type = 'smx'  # select from 'vq' and 'smx'
  kdq_d_in = 0  # when kdq_type == 'smx', we could reduce d_in
  kdq_share_subspace = True  # whether or not to share the subspace among D.
  additive_quantization = False
```











