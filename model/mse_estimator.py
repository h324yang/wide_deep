from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.contrib.learn import ModeKeys


_EPSILON = 1e-10


def build(model_dir, params={}, run_config=None):
  """Build an estimator appropriate for the given model type."""

  def model_fn(features, labels, mode, params):
    # wide model
    wide = tf.feature_column.input_layer(features, params["wide_columns"])

    # deep model
    deep = tf.feature_column.input_layer(features, params["deep_columns"])
    for num_unit in params["hidden_units"]:
        deep = tf.layers.dense(deep, num_unit, tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        # deep = tf.layers.dropout(deep, params["dropout_rate"], training=mode == ModeKeys.TRAIN)

    # wide + deep
    wide_deep = tf.concat([wide, deep], 1)
    logits = tf.layers.dense(wide_deep, params["n_class"])
    predictions = tf.reshape(logits, [-1]) # scores

    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(tf.cast(labels, tf.float32), predictions)
        optimizer = tf.train.ProximalAdagradOptimizer(params["learning_rate"], l1_regularization_strength=0.001)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())
        int_pred = tf.cast(tf.round(predictions), tf.int32)
        eval_metric_ops = { "accuracy": tf.metrics.accuracy(tf.cast(labels, tf.int32), int_pred) }

    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
  return tf.estimator.Estimator(model_fn, model_dir, run_config, params)


