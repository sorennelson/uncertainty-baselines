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

"""Kaggle diabetic retinopathy detection dataset builder."""

from typing import Dict, Optional

import tensorflow as tf
import tensorflow_addons.image as tfa_image
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base


class DiabeticRetinopathyDetectionDataset(base.BaseDataset):
  """Kaggle diabetic retinopathy detection dataset builder class."""

  def __init__(
      self,
      split: str,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = tf.data.experimental.AUTOTUNE,
      data_dir: Optional[str] = None,
      download_data: bool = False,
      is_training: Optional[bool] = None):
    """Create a Kaggle diabetic retinopathy detection tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      download_data: Whether or not to download data before loading.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    dataset_builder = tfds.builder(
        'diabetic_retinopathy_detection/btgraham-300', data_dir=data_dir)
    super().__init__(
        name='diabetic_retinopathy_detection',
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      
      if self.split == tfds.Split.TRAIN:
          image = example['image']
          image = tf.image.convert_image_dtype(image, tf.float32)

          # Inception Crop
          width = tf.cast(tf.shape(image),tf.float32)[1]
          height = tf.cast(tf.shape(image),tf.float32)[0]
          min_side = tf.minimum(width,height)

          crop_size =  tf.random.uniform([], minval=0.08, maxval=1.) * tf.cast(min_side, tf.float32)
          crop_size = tf.cast(crop_size, tf.int32)
          image = tf.image.random_crop(image, [crop_size,crop_size,3])
          image = tf.image.resize(image, [224,224])
          image = tf.image.random_flip_left_right(image)
      else:
          image = example['image']
          image = tf.image.convert_image_dtype(image, tf.float32)

          # Resize to 352 then crop to 320
          image = tf.image.resize(image, [352, 352])
          offset_width = (352 - 320) // 2
          offset_height = (352 - 320) // 2
          image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, 320,320)

      label = tf.one_hot(example['label'], 5, dtype=tf.float32)
      
      parsed_example = {
          'features': image,
          'labels': label,
          'name': example['name'],
      }
      return parsed_example

    return _example_parser