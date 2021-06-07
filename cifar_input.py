# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""CIFAR dataset input module.
"""

import tensorflow as tf
import os


def build_input(dataset, data_path, batch_size, mode, gray, classes):
    """Build CIFAR image and labels.

    Args:
      dataset: signlang.
      data_path: Filename for data.
      batch_size: Input batch size.
      mode: Either 'train' or 'eval'.
    Returns:
      images: Batches of images. [batch_size, image_size, image_size, channels]
      labels: Batches of labels. [batch_size, num_classes]
    Raises:
      ValueError: when the specified dataset is not supported.
    """
    image_size = 64  # target image size
    if dataset == 'signlang':
        num_classes = classes
    else:
        raise ValueError('Not supported dataset %s', dataset)
    depth = 3

    data_files = tf.gfile.Glob(os.path.join(data_path, "train*"))
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)

    if dataset == 'signlang':  # TFRecord format
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        assert len(image.shape) == 3
        label = tf.cast(features['image/class/label'], tf.int32)
        label = tf.reshape(label, [1])
        image = tf.cast(image, tf.float32)

        if mode == 'train' or mode == 'freeze':
            if gray:  # Gray color
                image = tf.image.rgb_to_grayscale(image)  # Gray color
                depth = 1

            # tf.image.decode_image returns RGB format, while we want to train w/ BGR (for FPGA HW). So, we convert
            # the order
            channels = tf.unstack(image, axis=-1)

            if gray:
                image = tf.stack([channels[0]], axis=-1)
            else:
                # RGB to BGR Conversion
                image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

            # image /= 128.0  # [0, 2)
            image = image - 128.0

            example_queue = tf.RandomShuffleQueue(
                capacity=16 * batch_size,
                min_after_dequeue=8 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 16
        else:  # evaluation
            if gray:  # Gray color
                image = tf.image.rgb_to_grayscale(image)  # Gray color
                depth = 1

            # tf.image.decode_image returns RGB format, while we want to train w/ BGR (for FPGA HW). So, we convert
            # the order
            channels = tf.unstack(image, axis=-1)
            if gray:
                image = tf.stack([channels[0]], axis=-1)
            else:
                # RGB to BGR conversion
                image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

            image -= 128.0

            example_queue = tf.FIFOQueue(
                3 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[image_size, image_size, depth], [1]])
            num_threads = 1

    example_enqueue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1),
        [batch_size, num_classes], 1.0, 0.0)

    assert len(images.get_shape()) == 4
    assert images.get_shape()[0] == batch_size
    assert images.get_shape()[-1] == depth
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer.
    channels = tf.unstack(images, axis=-1)  # images=BGR, images_viz=RGB
    if gray:
        images_viz = tf.stack([channels[0]], axis=-1)
    else:
        images_viz = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    tf.summary.image('images', images_viz+128)
    return images, labels
