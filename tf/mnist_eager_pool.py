# Copyright 2019 Lorenzo Bolzani. All Rights Reserved.
# Based on TensorFlow Authors samples
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
"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

This program demonstrates training of the convolutional neural network model
defined in mnist.py with eager execution enabled.

If you are not interested in eager execution, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf

#from official.mnist import dataset as mnist_dataset
#from keras.datasets import fashion_mnist as mnist_dataset

from official.mnist import mnist
from official.utils.flags import core as flags_core
from official.utils.misc import model_helpers

import numpy as np
import cv2
import random

tf.enable_eager_execution()
tfe = tf.contrib.eager


def set_seed(seed):
  print("SEED", seed)
  np.random.seed(seed)
  random.seed(seed)
  tf.random.set_random_seed(seed)


TRAIN_EPOCHS=45
FULL_BATCH_SIZE = 100
indexes = np.arange(FULL_BATCH_SIZE)
mnist_indexes = np.arange(60000)

retry_indexes = np.arange(100 * 1000)  # this depends on the update_losses_interval, just set it very large and we are safe

SEED=None

curiosity = None

RETRY_POOL_SIZE = None
retry_pool_sample_ratio = None
worst_data = None
full_sample = False

RETRY_SIZE = None   # this may be smaller in the initial iterations

squared = None


#retry_size = None
#MAX_RETRY_SIZE = None #int(BATCH_SIZE / 3)


update_losses_interval = 50

def normalize(probs):
    return probs / np.sum(probs)

pool = True

def loss(logits, labels, batch_data=None, training=True):

  #print("loss: logits, labels", logits[0], labels[0])
  global worst_data

  per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
  #print("per_sample", per_sample, labels)

  if training:
    
    k = int(round(FULL_BATCH_SIZE * curiosity_ratio))

    # If we use more samples from the pool than the ones "from mnist"
    k = min(len(per_sample), k)

    # we may not have enough elements in the pool
    retry_size = min(len(worst_data[2])+k, RETRY_SIZE)  # I will add k elements on this iteration
    #retry_size = k

    if curiosity:


      worst = np.argpartition(per_sample, -k)
      worst_idx = worst[-k:]
      retry_idx = worst_idx

      #print("loss: worst_idx", worst_idx[0])
      #print("loss: worst", per_sample[worst_idx[0]], "label", y_train[worst_idx[0]])

      #print(X_train[worst_idx[0]].shape)
      #cv2.imshow("worst", X_train[worst_idx[0]])
      #cv2.waitKey(0)

      #print("worst k", worst_idx)
      if pool:

        #print("Append", len(retry_idx))

        worst_data[0].extend(batch_data[0][retry_idx])
        worst_data[1].extend(batch_data[1][retry_idx])
        #print(worst_idx)
        #print("len", len(worst_data[0]), len(worst_data[1]))
        #print(per_sample[13])
        for i in range(len(worst_idx)):
          #print(worst_idx[i])
          elem = per_sample[worst_idx[i]].numpy()
          worst_data[2].append(elem)

        # :(
        x_worst = np.asarray(worst_data[0], dtype='float32')
        y_worst = np.asarray(worst_data[1], dtype='int32')
        loss_worst = np.asarray(worst_data[2])


        # sample su tutto il pool
        if full_sample:
          pow_loss_worst = loss_worst ** squared
          priorities = normalize(pow_loss_worst)
          retry_idx = np.random.choice(retry_indexes[:len(priorities)], size=retry_size, replace=False, p=priorities)
        else:
          # subset sample
          pool_sample_size = int(round(RETRY_POOL_SIZE * retry_pool_sample_ratio))
          pool_sample_size = min(len(loss_worst), pool_sample_size)
          
          pool_worst = np.argpartition(loss_worst, -pool_sample_size)
          pool_worst_idx = pool_worst[-pool_sample_size:]
          
          worst_subset = loss_worst[pool_worst_idx]
          worst_priorities = normalize(worst_subset)

          retry_idx = np.random.choice(pool_worst_idx, size=retry_size, replace=False, p=worst_priorities)

        retry_images = x_worst[retry_idx]
        retry_labels = y_worst[retry_idx]
      else:
        retry_images = batch_data[0][retry_idx]
        retry_labels = batch_data[1][retry_idx]

    else:
      if pool:
        # sample from all MNIST
        retry_idx = np.random.choice(mnist_indexes, size=retry_size, replace=False)
        retry_images = X_train[retry_idx]
        retry_labels = y_train[retry_idx]
      else:
        retry_idx = np.random.choice(indexes[:BATCH_SIZE], size=k, replace=False)
        retry_images = batch_data[0][retry_idx]
        retry_labels = batch_data[1][retry_idx]

    #print("num retry_images", len(retry_images))
    return tf.reduce_mean(per_sample), retry_images, retry_labels, per_sample
  
  return tf.reduce_mean(per_sample), None, None, per_sample


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.cast(labels, tf.int64)
  batch_size = int(logits.shape[0])
  return tf.reduce_sum(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def data_generator_mnist(X_test, y_test, batchSize):

    #if(isTrain):
    #    dataset = (X_train, y_train.astype('int32'))
    #    dataset_size = len(X_train)
    #else :

    dataset = (X_test, y_test)
    dataset_size = len(X_test)

    #print("dataset_size", dataset_size)
    #print("batchSize", batchSize)

    i = 0
    while(True):
        #print("i", i)
        if (i+batchSize > dataset_size):
            #print("i loop", i)
            head = dataset[0][i:], dataset[1][i:]
            #print("head", head[0].shape)
            rest = batchSize - head[0].shape[0]
            #print("rest", rest)
            tail = dataset[0][:rest], dataset[1][:rest]
            #print("tail", tail[0].shape)
            yield np.concatenate((head[0], tail[0])), np.concatenate((head[1], tail[1]))
            i = rest
        else:
          yield dataset[0][i:i+batchSize], dataset[1][i:i+batchSize]
          #print("yield from", i, "to", i+batchSize)
          i += batchSize
          #print("i post", i)


import keras

nb_classes = 10
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
#(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#worst_idx = 10000

data_gen = None
test_gen = None

def train(model, optimizer, _, step_counter, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""

  global worst_data

  start = time.time()
  #for (batch, (images, labels)) in enumerate(dataset):
  for batch in range(1000):

    #idx = np.arange(50)
    #images = tf.constant(X_train[idx], dtype=np.float32)
    #labels = tf.constant(y_train[idx], dtype=np.int32)
    batch_data = next(data_gen)
    images = batch_data[0]
    labels = batch_data[1]

    with tf.contrib.summary.record_summaries_every_n_global_steps(
        10, global_step=step_counter):
      # Record the operations used to compute the loss given the input,
      # so that the gradient of the loss with respect to the variables
      # can be computed.
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss_value, retry_images, retry_labels, _ = loss(logits, labels, batch_data)

        #check_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        #print("check_loss", check_loss[worst_idx])

        #images = tf.constant([X_train[worst_idx]], dtype=np.float32)
        #labels = tf.constant([y_train[worst_idx]], dtype=np.int32)
        #logits = model(images, training=False)
        #print("logits 2", logits)
        #check_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        #print("check_loss bis", check_loss[0], "label", y_train[worst_idx])

        #cv2.imshow("check img", X_train[worst_idx])
        #cv2.waitKey(0)

        tf.contrib.summary.scalar('loss', loss_value)
        tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))

      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          zip(grads, model.variables), global_step=step_counter)

      # retry
      #print("len", retry_images.shape)
      with tf.GradientTape() as tape:
        logits = model(retry_images, training=True)
        #print("logits 1", logits)
        loss_value, *_ = loss(logits, retry_labels, training=False)

      grads = tape.gradient(loss_value, model.variables)
      optimizer.apply_gradients(
          zip(grads, model.variables), global_step=step_counter)

      # TODO: update worst_loss data for the retry data

      if pool:
        if batch % update_losses_interval == 0 and len(worst_data[2]) > 0:
          print("Updating losses...", np.mean(worst_data[2]), len(worst_data[2]))
          curr_loss = np.mean(worst_data[2])
          images = np.asarray(worst_data[0], dtype='float32').reshape(len(worst_data[0]), 28, 28, 1)
          labels = worst_data[1]
          logits = model(images, training=False)
          _, _, _, all_losses = loss(logits, labels, batch_data, training=False)
          worst_data[2] = all_losses.numpy().tolist()

          if len(worst_data[2]) > RETRY_POOL_SIZE:
            # discard best
            #print("Discard best matches")

            # I should just discard the oldest...

            losses = np.asarray(worst_data[2])
            idx = np.argsort(losses, axis=0)
            new = []
            for i in idx[-RETRY_POOL_SIZE:]:
              new.append(worst_data[0][i])
            worst_data[0] = new.copy()

            new = []
            for i in idx[-RETRY_POOL_SIZE:]:
              new.append(worst_data[1][i])
            worst_data[1] = new.copy()

            new = []
            for i in idx[-RETRY_POOL_SIZE:]:
              new.append(worst_data[2][i])
            worst_data[2] = new.copy()

            '''
            img = np.asarray(worst_data[0].copy())
            labels = np.asarray(worst_data[1].copy())
            losses = losses[idx]
            labels = labels[idx]
            img = img[idx]


            worst_data[0] = list(img.tolist())
            worst_data[1] = list(labels.tolist())
            worst_data[2] = list(losses.tolist())
            
            #gc.collect()

            #print(idx)
            '''

            '''

            print(worst_data[2][idx[0]])
            print(worst_data[2][idx[-1]])
            print(idx[0])

            print(idx[RETRY_POOL_SIZE-len(worst_data[2]):])

            best = idx[:len(worst_data[2])-RETRY_POOL_SIZE]
            print(worst_data[0][0].shape)
            worst_data[0] = np.delete(worst_data[0], best).tolist()
            print(worst_data[0][0].shape)
            worst_data[1] = np.delete(worst_data[1], best).tolist()
            worst_data[2] = np.delete(worst_data[2], best).tolist()
            print(type(worst_data[2]))
            '''

            '''
            worst_data[0] = np.asarray(worst_data[0].copy())[idx].tolist()
            worst_data[1] = np.asarray(worst_data[1].copy())[idx].tolist()
            worst_data[2] = np.asarray(worst_data[2].copy())[idx].tolist()

            '''
            '''
            worst_data[0] = worst_data[0][-RETRY_POOL_SIZE:]
            worst_data[1] = worst_data[1][-RETRY_POOL_SIZE:]
            worst_data[2] = worst_data[2][-RETRY_POOL_SIZE:]
            '''

            #worst_data[0], worst_data[1], worst_data[2] = zip(*sorted(zip(worst_data[0], worst_data[1], worst_data[2])))

            #print(worst_data[2])
            #print(len(worst_data[2]))

            #exit(1)

          print("Loss udpate done", curr_loss, "=>", np.mean(worst_data[2]), len(worst_data[2]))

      if log_interval and batch % log_interval == 0:
        rate = log_interval / (time.time() - start)
        print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
        start = time.time()

validation=[]

def test(model, _, epoch):

  """Perform an evaluation of `model` on the examples from `dataset`."""
  avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
  accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)

  #for (images, labels) in dataset:
  for batch in range(1000):

    batch_data = next(test_gen)
    images = batch_data[0]
    labels = batch_data[1]

    logits = model(images, training=False)
    avg_loss(loss(logits, labels, training=False)[0])
    accuracy(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
  print('Test set(%s): Average loss: %.4f, Accuracy: %4f%%\n' %
        (epoch, avg_loss.result(), 100 * accuracy.result()))
  with tf.contrib.summary.always_record_summaries():
    tf.contrib.summary.scalar('loss', avg_loss.result())
    tf.contrib.summary.scalar('accuracy', accuracy.result())

  validation.append(accuracy.result().numpy())

def run_mnist_eager(flags_obj):
  """Run MNIST training and eval loop in eager mode.

  Args:
    flags_obj: An object containing parsed flag values.
  """

  model_helpers.apply_clean(flags.FLAGS)

  # Automatically determine device and data_format
  (device, data_format) = ('/gpu:0', 'channels_first')
  if flags_obj.no_gpu or not tf.test.is_gpu_available():
    (device, data_format) = ('/cpu:0', 'channels_last')
  # If data_format is defined in FLAGS, overwrite automatically set value.
  if flags_obj.data_format is not None:
    data_format = flags_obj.data_format
  print('Using device %s, and data format %s.' % (device, data_format))

  # Load the datasets
  #train_ds = mnist_dataset.train(flags_obj.data_dir).shuffle(60000).batch(
  #    flags_obj.batch_size)
  #test_ds = mnist_dataset.test(flags_obj.data_dir).batch(
  #    flags_obj.batch_size)

  # Create the model and optimizer
  model = mnist.create_model(data_format)
  optimizer = tf.train.MomentumOptimizer(flags_obj.lr, flags_obj.momentum)

  # Create file writers for writing TensorBoard summaries.
  if flags_obj.output_dir:
    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(flags_obj.output_dir, 'train')
    test_dir = os.path.join(flags_obj.output_dir, 'eval')
    tf.gfile.MakeDirs(flags_obj.output_dir)
  else:
    train_dir = None
    test_dir = None
  summary_writer = tf.contrib.summary.create_file_writer(
      train_dir, flush_millis=10000)
  test_summary_writer = tf.contrib.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')

  # Create and restore checkpoint (if one exists on the path)
  checkpoint_prefix = os.path.join(flags_obj.model_dir, 'ckpt')
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tf.train.Checkpoint(
      model=model, optimizer=optimizer, step_counter=step_counter)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(flags_obj.model_dir))

  # Train and evaluate for a set number of epochs.
  with tf.device(device):
    for epoch in range(flags_obj.train_epochs):
      start = time.time()
      with summary_writer.as_default():
        train_ds = None
        train(model, optimizer, train_ds, step_counter, flags_obj.log_interval)



      end = time.time()
      print('\nTrain time for epoch #%d (%d total steps): %f' %
            (epoch + 1,
             step_counter.numpy(),
             end - start))
      with test_summary_writer.as_default():
        test(model, None, epoch + 1)
      print(checkpoint_prefix)
      checkpoint.save(checkpoint_prefix)


def define_mnist_eager_flags():
  """Defined flags and defaults for MNIST in eager mode."""
  flags_core.define_base_eager()
  flags_core.define_image()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_integer(
      name='log_interval', short_name='li', default=10,
      help=flags_core.help_wrap('batches between logging training status'))

  flags.DEFINE_string(
      name='output_dir', short_name='od', default=sys.argv[1],
      help=flags_core.help_wrap('Directory to write TensorBoard summaries'))

  flags.DEFINE_float(name='learning_rate', short_name='lr', default=0.01,
                     help=flags_core.help_wrap('Learning rate.'))

  flags.DEFINE_float(name='momentum', short_name='m', default=0.5,
                     help=flags_core.help_wrap('SGD momentum.'))

  flags.DEFINE_bool(name='no_gpu', short_name='nogpu', default=False,
                    help=flags_core.help_wrap(
                        'disables GPU usage even if a GPU is available'))

  flags_core.set_defaults(
      data_dir='/tmp/tensorflow/mnist/input_data',
      model_dir='/tmp/tensorflow/mnist/checkpoints/',
      batch_size=100,
      train_epochs=TRAIN_EPOCHS,
  )

from random import randint
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

def main(_):

  global pool, curiosity, curiosity_ratio, RETRY_SIZE, RETRY_POOL_SIZE
  global worst_data, data_gen, test_gen, BATCH_SIZE, squared, retry_pool_sample_ratio

  MEAN_ITER = 1

  fig, ax = plt.subplots()
  ax.grid()
  t = np.arange(0, TRAIN_EPOCHS)
  all_validation = []

  colors = ['b', 'r', 'g', 'm', 'c', 'k', 'y', 'b--', 'r--', 'g--']

  '''
  cur_ratios = [0.2, 0.2, 0.2, 0.2, 0.2]
  pool_val =  [True, True, True, True, True]
  cur_val =    [True, True, True, True, True]
  pool_size =  [50, 100, 250, 500, 1000]
  retry_size_ratios = [0.2, 0.2, 0.2, 0.2, 0.2]  # TODO
  '''
  cur_ratios = [0.33]
  pool_val =  [False]
  cur_val =    [True]
  pool_size =  [1000]
  retry_size_ratios = [0.33]  # TODO
  retry_pool_sample_ratio = 0.3

  powers = [1, 2, 3, 4]
  for ci, cr in enumerate(cur_ratios):
    #curiosity = ci % 2 != 0

    #squared = ci % 2 != 0
    squared = powers[ci]

    pool = pool_val[ci]
    curiosity_ratio = cr
    curiosity = cur_val[ci]
    RETRY_POOL_SIZE = pool_size[ci]

    #RETRY_SIZE = int(round(FULL_BATCH_SIZE / curiosity_ratio))    # TEMP: questo e' libero!
    RETRY_SIZE = int(round(FULL_BATCH_SIZE * retry_size_ratios[ci]))

    #k = int(round(FULL_BATCH_SIZE * curiosity_ratio))   # extra values

    print("## Run: ratio", cr, "pool", pool)
    for i in range(MEAN_ITER):
      print("## Iteration", i, "/", MEAN_ITER, " - ratio", cr)
      set_seed(randint(0, 100000))
      
      worst_data = [[], [], []]

      #retry_size = min(len(worst_data[2])+k, RETRY_SIZE)
      BATCH_SIZE = FULL_BATCH_SIZE - RETRY_SIZE
      print("Batch size:", BATCH_SIZE, "retry size:", RETRY_SIZE)

      data_gen = data_generator_mnist(X_train, y_train, BATCH_SIZE)
      test_gen = data_generator_mnist(X_test, y_test, BATCH_SIZE)

      run_mnist_eager(flags.FLAGS)
      print("acc", validation)

      ax.plot(t, validation, colors[ci], alpha=0.1)
      fig.savefig("all.png", dpi=200)

      all_validation.append(validation.copy())
      validation.clear()
    c_mean = np.mean(all_validation, axis=0)
    ax.plot(t, c_mean, colors[ci])
    fig.savefig("all.png", dpi=200)
    all_validation.clear()

  fig.savefig("all_" + str(time.time()) + ".png", dpi=200)


  #plt.show()



if __name__ == '__main__':
  define_mnist_eager_flags()
  absl_app.run(main=main)
_
