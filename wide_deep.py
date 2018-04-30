from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order
from model import focal_estimator, xent_estimator, mse_estimator, metrics, loader
from tensorflow.contrib.learn import ModeKeys
from official.utils.arg_parsers import parsers
# from official.utils.logs import hooks_helper
# from official.utils.misc import model_helpers


def main(argv):
  parser = WideDeepArgParser()
  flags = parser.parse_args(args=argv[1:])

  with open(flags.params) as f:
      params = json.load(f)

  train_file = os.path.join(flags.data_dir, 'train.csv')
  test_file = os.path.join(flags.data_dir, 'test.csv')
  if 'alpha' in params and len(params['alpha']) == 0:
      params['alpha'] = loader.calculate_alpha(test_file, 5)
  print("model hyper-parameters", sorted(params))

  model_dir = os.path.join(flags.model_dir, flags.name)
  # Clean up the model directory if present
  if flags.mode == 'retrain':
      shutil.rmtree(model_dir, ignore_errors=True)

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  if flags.device == 'gpu':
      run_config = tf.estimator.RunConfig()
  else:
      run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))

  # Load data
  wide_columns, deep_columns = loader.build_model_columns()
  params["wide_columns"] = wide_columns
  params["deep_columns"] = deep_columns

  # Select model.
  if flags.loss == 'focal':
      model = focal_estimator.build(model_dir, params, run_config)
  elif flags.loss == 'xent':
      model = xent_estimator.build(model_dir, params, run_config)
  elif flags.loss == 'mse':
      model = mse_estimator.build(model_dir, params, run_config)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return loader.input_fn(
        train_file, flags.epochs_between_evals, True, flags.batch_size)

  def eval_input_fn():
    return loader.input_fn(test_file, 1, False, flags.batch_size)

  def pred_input_fn():
    return loader.input_fn(test_file, 1, False, flags.batch_size, True)


  # loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
  # train_hooks = hooks_helper.get_train_hooks(
      # flags.hooks, batch_size=flags.batch_size,
      # tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      # 'loss': loss_prefix + 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def eval_ndcg(topks=[10]):
    scores = [s for s in model.predict(input_fn=pred_input_fn)]
    labels = loader.get_labels(test_file)
    ranked = labels[np.argsort(scores)[::-1]]
    for topk in topks:
        print("ndcg@%s: %s"%(topk, metrics.ndcg_at_k(ranked, topk)))
    return True

  topks = [10, 50, 100]
  if flags.mode == 'ndcg':
        # Accuracy
        results = model.evaluate(input_fn=eval_input_fn)
        for key in sorted(results):
          print('%s: %s' % (key, results[key]))
        # NDCG
        eval_ndcg(topks)
  else:
    for n in range(flags.train_epochs // flags.epochs_between_evals):
        # model.train(input_fn=train_input_fn, hooks=train_hooks)
        model.train(input_fn=train_input_fn)

        # Display evaluation metrics
        print('-' * 60)
        print('Results at epoch', (n + 1) * flags.epochs_between_evals)
        # Accuracy
        results = model.evaluate(input_fn=eval_input_fn)
        for key in sorted(results):
          print('%s: %s' % (key, results[key]))
        # NDCG
        eval_ndcg(topks)


class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()])
    self.add_argument(
        '--mode', '-m', type=str, default='ndcg',
        choices=['train', 'retrain', 'ndcg'],
        help='[default: %(default)s] Model types: train, retrain, ndcg.',
        metavar='<M>')

    self.add_argument(
        '--params', '-p', type=str, default='configs/std_params.json',
        help='[default: %(default)s] Hyper-parameter setting: a json object.',
        metavar='<P>')

    self.add_argument(
        '--name', '-n', type=str, default='tmp_model',
        help='[default: %(default)s] Model name.',
        metavar='<N>')

    self.add_argument(
        '--loss', '-l', type=str, default='focal',
        choices=['focal', 'xent', 'mse'],
        help='[default: %(default)s] Model loss.',
        metavar='<L>')

    self.add_argument(
        '--device', '-d', type=str, default='gpu',
        choices=['cpu', 'gpu'],
        help='[default: %(default)s] Select device.',
        metavar='<d>')

    self.set_defaults(
        data_dir='./data/',
        model_dir='./model_saved/',
        train_epochs=100,
        epochs_between_evals=1,
        batch_size=64)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
