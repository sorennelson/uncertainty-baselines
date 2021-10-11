# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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


import os
import time
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp

from classification_models.tfkeras import Classifiers


flags.DEFINE_float('label_smoothing', 0., 'Label smoothing parameter in [0,1].')
flags.register_validator('label_smoothing',
                         lambda ls: ls >= 0.0 and ls <= 1.0,
                         message='--label_smoothing must be in [0, 1].')

# Data Augmentation flags.
flags.DEFINE_bool('augmix', False,
                  'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer('aug_count', 1,
                     'Number of augmentation operations in AugMix to perform '
                     'on the input image. In the simgle model context, it'
                     'should be 1. In the ensembles context, it should be'
                     'ensemble_size if we perform random_augment only; It'
                     'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth. This corresponds'
                     'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer('augmix_width', 3,
                     'Augmix width. This corresponds to the k in line 5 in the'
                     'Algorithm box in [4].')

# Fine-grained specification of the hyperparameters (used when FLAGS.l2 is None)
flags.DEFINE_float('bn_l2', None, 'L2 reg. coefficient for batch-norm layers.')
flags.DEFINE_float('input_conv_l2', None,
                   'L2 reg. coefficient for the input conv layer.')
flags.DEFINE_float('group_1_conv_l2', None,
                   'L2 reg. coefficient for the 1st group of conv layers.')
flags.DEFINE_float('group_2_conv_l2', None,
                   'L2 reg. coefficient for the 2nd group of conv layers.')
flags.DEFINE_float('group_3_conv_l2', None,
                   'L2 reg. coefficient for the 3rd group of conv layers.')
flags.DEFINE_float('dense_kernel_l2', None,
                   'L2 reg. coefficient for the kernel of the dense layer.')
flags.DEFINE_float('dense_bias_l2', None,
                   'L2 reg. coefficient for the bias of the dense layer.')


flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')

flags.DEFINE_bool('resnet_18', False, 'Whether to use ResNet 18')
flags.DEFINE_bool('pretrained_resnet_18', False, 'Whether to use Pretrained ResNet 18')
flags.DEFINE_float('reset_prob', 0.0, 'Whether to reset weights with some probability')

flags.DEFINE_bool('limit_memory', False, 'Whether to limit GPU memory.')
flags.DEFINE_integer('reset_stage', None, 'Which block to reset')
flags.DEFINE_integer('final_layer_reset', None, 'Resetting final layers')
flags.DEFINE_integer('unit_2_reset', None, 'How many final units to reset.')


FLAGS = flags.FLAGS



def reinitialize_model(model, probability=0.0, reset_stage_idx=None, final_layer_reset=None, unit_2_reset=None):
  residual_blocks = ['stage1_unit1','stage1_unit2','stage2_unit1','stage2_unit2','stage3_unit1','stage3_unit2','stage4_unit1','stage4_unit2' ]
  if reset_stage_idx is not None:
      residual_blocks = residual_blocks[reset_stage_idx:reset_stage_idx + 1]
  if final_layer_reset is not None:
      residual_blocks = residual_blocks[-1*final_layer_reset:]
  if unit_2_reset is not None:
      residual_blocks =  ['stage1_unit2','stage2_unit2','stage3_unit2','stage4_unit2' ]
      random.shuffle(residual_blocks)
      residual_blocks = residual_blocks[:unit_2_reset]
  initializers = []
  weights = []
  for block in residual_blocks:
    if np.random.random() >= probability:
      pass
    else:
      for layer in model.layers:
        if block in layer.name:
          if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            weights += [layer.kernel]
            initializers += [layer.kernel_initializer]
            if layer.bias is not None:
              weights += [layer.bias]
              initializers += [layer.bias_initializer]
          elif isinstance(layer, tf.keras.layers.BatchNormalization):
            weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
            initializers += [layer.gamma_initializer,
                            layer.beta_initializer,
                            layer.moving_mean_initializer,
                            layer.moving_variance_initializer]
  for w, init in zip(weights, initializers):
    w.assign(init(w.shape, dtype=w.dtype))

def _extract_hyperparameter_dictionary():
  """Create the dictionary of hyperparameters from FLAGS."""
  flags_as_dict = FLAGS.flag_values_dict()
  hp_keys = ub.models.models.wide_resnet.HP_KEYS
  hps = {k: flags_as_dict[k] for k in hp_keys}
  return hps


def _generalized_energy_distance(labels, predictions, num_classes):
  """Compute generalized energy distance.

  See Eq. (8) https://arxiv.org/abs/2006.06015
  where d(a, b) = (a - b)^2.

  Args:
    labels: [batch_size, num_classes] Tensor with empirical probabilities of
      each class assigned by the labellers.
    predictions: [batch_size, num_classes] Tensor of predicted probabilities.
    num_classes: Integer.

  Returns:
    Tuple of Tensors (label_diversity, sample_diversity, ged).
  """
  y = tf.expand_dims(labels, -1)
  y_hat = tf.expand_dims(predictions, -1)

  non_diag = tf.expand_dims(1.0 - tf.eye(num_classes), 0)
  distance = tf.reduce_sum(tf.reduce_sum(
      non_diag * y * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
  label_diversity = tf.reduce_sum(tf.reduce_sum(
      non_diag * y * tf.transpose(y, perm=[0, 2, 1]), -1), -1)
  sample_diversity = tf.reduce_sum(tf.reduce_sum(
      non_diag * y_hat * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
  ged = tf.reduce_mean(2 * distance - label_diversity - sample_diversity)
  return label_diversity, sample_diversity, ged


def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv  # unused arg

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)
  physical_devices = tf.config.list_physical_devices('GPU')
  if FLAGS.limit_memory:
     try:
        tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limt= 12288)])
        tf.config.experimental.set_virtual_device_configuration(physical_devices[1],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limt= 12288)])
     except:
        pass
  else:
     try:
       tf.config.experimental.set_memory_growth(physical_devices[0], True)
       tf.config.experimental.set_memory_growth(physical_devices[1], True)
     except:
       # Invalid device or cannot modify virtual devices once initialized.
       pass

  data_dir = FLAGS.data_dir
  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  train_dataset_size = 120905
  test_dataset_size = 52041
  #steps_per_epoch = 15
  steps_per_epoch = int(train_dataset_size / batch_size)
  logging.info('Steps per epoch %s', steps_per_epoch)
  logging.info('Size of the dataset %s', train_dataset_size)
  logging.info('Train proportion %s', FLAGS.train_proportion)
  steps_per_eval = test_dataset_size // batch_size

  aug_params = {
      'augmix': FLAGS.augmix,
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth,
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width,
  }

  # Note that stateless_{fold_in,split} may incur a performance cost, but a
  # quick side-by-side test seemed to imply this was minimal.
  seeds = tf.random.experimental.stateless_split(
      [FLAGS.seed, FLAGS.seed + 1], 2)[:, 0]
  train_builder = ub.datasets.get(
      'domainnet',
      domain='real',
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      shuffle_buffer_size=1000,
      split=tfds.Split.TRAIN)
  train_dataset = train_builder.load(batch_size=batch_size)
  validation_dataset = None
  steps_per_validation = steps_per_eval
  clean_test_builder = ub.datasets.get(
      'domainnet',
      domain='real',
      split=tfds.Split.TEST,
      data_dir=data_dir)
  clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }


  train_dataset = strategy.experimental_distribute_dataset(train_dataset)


  steps_per_epoch = train_dataset_size // batch_size
  steps_per_eval = test_dataset_size // batch_size
  num_classes = 345


  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model')
    if FLAGS.resnet_18:
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        inputs = tf.keras.layers.Input((224,224,3))
        resnet_trunk = ResNet18((224, 224, 3), l2=FLAGS.l2, weights=None, include_top=False)(inputs)
        pooled_resnet = tf.keras.layers.GlobalAveragePooling2D()(resnet_trunk)
        output = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2),
                bias_regularizer=tf.keras.regularizers.l2(FLAGS.l2))(pooled_resnet)
        model = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet_18')
    elif FLAGS.pretrained_resnet_18:
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        inputs = tf.keras.layers.Input((224,224,3))
        resnet_trunk = ResNet18((224, 224, 3), l2=FLAGS.l2, weights='imagenet', include_top=False)
        if FLAGS.reset_prob > 0.0:
            reinitialize_model(resnet_trunk, probability = FLAGS.reset_prob, reset_stage_idx=FLAGS.reset_stage, final_layer_reset=FLAGS.final_layer_reset, unit_2_reset=FLAGS.unit_2_reset)
        resnet_trunk_out = resnet_trunk(inputs)
        pooled_resnet = tf.keras.layers.GlobalAveragePooling2D()(resnet_trunk_out)
        output = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2),
                bias_regularizer=tf.keras.regularizers.l2(FLAGS.l2))(pooled_resnet)
        model = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet_18')
    else:
        model = ub.models.wide_resnet(
            input_shape=(224, 224, 3),
            depth=28,
            width_multiplier=10,
            num_classes=num_classes,
            l2=FLAGS.l2,
            hps=_extract_hyperparameter_dictionary(),
            seed=seeds[1])

    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=True)
    metrics = {
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss':
            tf.keras.metrics.Mean(),
        'train/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
    }
    if validation_dataset:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece': rm.metrics.ExpectedCalibrationError(
              num_bins=FLAGS.num_bins),
      })

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      if FLAGS.augmix and FLAGS.aug_count >= 1:
        # Index 0 at augmix processing is the unperturbed image.
        # We take just 1 augmented image from the returned augmented images.
        images = images[:, 1, ...]
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.label_smoothing == 0.:
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                              logits,
                                                              from_logits=True))
        else:
          one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.categorical_crossentropy(
                  one_hot_labels,
                  logits,
                  from_logits=True,
                  label_smoothing=FLAGS.label_smoothing))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))
      print('step taken')

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      logits = model(images, training=False)
      probs = tf.nn.softmax(logits)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
        metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
      else:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)

    for i in tf.range(tf.cast(num_steps, tf.int32)):
      print(i)
      strategy.run(step_fn, args=(next(iterator),))


  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
  metrics.update({'train/ms_per_example': tf.keras.metrics.Mean()})

  train_iterator = iter(train_dataset)
  start_time = time.time()
  tb_callback = None
  if FLAGS.collect_profile:
    tb_callback = tf.keras.callbacks.TensorBoard(
        profile_batch=(100, 102),
        log_dir=os.path.join(FLAGS.output_dir, 'logs'))
    tb_callback.set_model(model)
  #for epoch in range(1):
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)
    train_start_time = time.time()
    train_step(train_iterator)
    ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size
    metrics['train/ms_per_example'].update_state(ms_per_example)

    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * FLAGS.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps,
                   epoch + 1,
                   FLAGS.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
    logging.info(message)
    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    if validation_dataset:
      validation_iterator = iter(validation_dataset)
      test_step(
          validation_iterator, 'validation', 'clean', steps_per_validation)
    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      logging.info('Starting to run eval at epoch: %s', epoch)
      test_start_time = time.time()
      test_step(test_iterator, 'test', dataset_name, steps_per_eval)
      ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
      metrics['test/ms_per_example'].update_state(ms_per_example)

      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    total_results = {name: metric.result() for name, metric in metrics.items()}
    total_results.update(corrupt_results)
    # Metrics from Robustness Metrics (like ECE) will return a dict with a
    # single key/value, instead of a scalar.
    total_results = {
        k: (list(v.values())[0] if isinstance(v, dict) else v)
        for k, v in total_results.items()
    }
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
    })


if __name__ == '__main__':
  app.run(main)
