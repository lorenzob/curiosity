
# based on: https://keras.io/examples/mnist_cnn/

"""

There are many modes so this can be confusing.

Main ones are the following. All the following modes call fit two times with batches with the same size (batch_size
on the first call, k for the second). So these should be properly comparable.

CURIOSITY_MODE
This is the one described in the article. Pick a batch, train, compute losses, retrain on the difficult subset

CURIOSITY_BASELINE and CURIOSITY_BASELINE_FULL_SAMPLE
These are baselines, "classic" training. They call fit twice for proper comparison. First one imitates
CURIOSITY_MODE and randomly samples from the same batch, the second one randomly sample from the whole
dataset (it is a better comparison for the following modes).
CURIOSITY_BASELINE_FULL_SAMPLE is the proper "classic" training (repeated two times).

WEIGHTS_MODE
It uses the samples weights. It picks a "normal" batch, compute the losses, and map the losses to samples
weights and fits the batch with these params. It is repeated two times for easy comparison.

POOL_MODE
It keeps a pool of the most difficult samples encountered in each batch. It fits once on a "classic" batch
and once on a difficult batch randomly sampled from the pool.

ALL_POOL_MODE
Treats the whole dataset as a pool. Randomly pick a very large subset of the dataset (to avoid OOM and to speed up
computation, ideally it is the whole dataset). From this one selects the most difficult subset and fits on this.
It is repeated two times for easy comparison.

CURIOSITY_ALL_POOL_MODE
Fit a first normal batch than a hard one samples from the whole dataset. The first batch can be made a little harder
using the extra_sampling param.


The following mode calls fit multiple times so it is trickier to compare properly.

ITER_MODE
This one picks a large batch (i.e. 500 elements) and from this select most difficult subset and fit on this.
Repeat this multiple times. It tries to squeeze as much as possible from this large batch before moving to
the next one.
This mode is comparable only on by_sample_count, not by iterations. It is comparable only with curiosity_ratio = 1


Two charts are created, the "_by_samples" ones are the easiest to compare. Charts are strictly comparable only
for the same curiosity_ratio (and for modes with the same number of fit calls).

Unless I'm testing the CURIOSITY_MODE I prefer to use curiosity_ratio=1 so that the two fit calls use batches of
the same size.

"""


import os
import random
import shutil
import sys
import time
from builtins import property

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, MaxPool2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, Input

import numpy as np

import tensorflow as tf

num_classes = 10
# default: 3, 2
# big comparison: 6, 3
epochs = 100
average_count = 3

VALIDATION_ITERATIONS=100

EARLY_STOP_PATIENCE = 50
EARLY_STOP_TOLERANCE = 0.01 / 100

# debug settings
quick_debug=False
if quick_debug:
    epochs = 1
    average_count = 2
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_TOLERANCE = 1 / 100


SEED=123

DEFAULT_EXTRA_SAMPLING=0

shuffle = True

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)

name = sys.argv[1]

root=f"data/{name}"
os.makedirs(root, exist_ok=True)

train_callbacks=[]

def fprint(*args):
    print(*args)
    print(*args, file=open(f'{root}/log.txt', 'a'))

def data_generator_mnist(X, y, batchSize):

    if shuffle:
        #combined = list(zip(X, y))
        #random.shuffle(combined)
        #X[:], y[:] = zip(*combined)
        import sklearn
        X, y = sklearn.utils.shuffle(X, y)

    dataset = (X, y)
    dataset_size = len(X)

    i = 0
    while (True):
        if (i + batchSize > dataset_size):

            head = dataset[0][i:], dataset[1][i:]
            rest = batchSize - head[0].shape[0]
            tail = dataset[0][:rest], dataset[1][:rest]
            yield np.concatenate((head[0], tail[0])), np.concatenate((head[1], tail[1]))
            i = rest
        else:
            yield dataset[0][i:i + batchSize], dataset[1][i:i + batchSize]
            i += batchSize


# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
use_mnist = True
if use_mnist:
    fprint("Dataset MNIST")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    fprint("Dataset Fashion MNIST")
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# debug mode
x_train = x_train
y_train = y_train

def create_model():

    fprint("Base model")

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()

    return model

#https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
def create_adv_model():

    # also try:
    # https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1

    fprint("Advanced model")

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                     activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    model.summary()

    return model

def scale(value, fromMin, fromMax, toMin, toMax):

    fromSpan = fromMax - fromMin
    toSpan = toMax - toMin

    valueScaled = (value - fromMin) / fromSpan

    return toMin + (valueScaled * toSpan)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


BASELINE = 'BASELINE'
WEIGHTS_MODE = 'WEIGHTS_MODE'  # keras sample weights based on loss
CURIOSITY_MODE = 'CURIOSITY_MODE'
CURIOSITY_BASELINE = 'CURIOSITY_BASELINE'
CURIOSITY_BASELINE_FULL_SAMPLE = 'CURIOSITY_BASELINE_FULL_SAMPLE'
CURIOSITY_MODE_SINGLE_BATCH = 'CURIOSITY_MODE_SINGLE_BATCH'
POOL_MODE = 'POOL_MODE'
ITER_MODE = 'ITER_MODE'
ALL_POOL_MODE = 'ALL_POOL_MODE'
CURIOSITY_ALL_POOL_MODE = 'CURIOSITY_ALL_POOL_MODE'

def train(mode, base_batch_size, curiosity_ratio=1, params=None):

    print(f"Mode {mode}, curiosity_ratio: {curiosity_ratio}")

    pool_images = None
    pool_labels = None

    k = int(base_batch_size * curiosity_ratio)

    if mode in [BASELINE, WEIGHTS_MODE]:
        batch_size = base_batch_size + k
    elif mode in [ITER_MODE]:
        batch_size = base_batch_size * params['ratio']    # comparable only with cr 1
    else:
        batch_size = base_batch_size

    data_gen = data_generator_mnist(x_train, y_train, batch_size)

    warmup_iter = params['warmup_iter'] if 'warmup_iter' in params else 0
    print("warmup_iter", warmup_iter)

    validation = []
    validation_by_sample_count = []
    best_accuracies = []
    sample_count = 0
    train_start = time.time()
    iteration = 0   # number of fit call (note: some iterations process smaller batches than others)
    iter_with_no_improvements = 0
    best_acc = 0.0001    # very small numeber
    best_acc_iter = 0
    absolute_best_acc = 0    # very small number
    for e in range(epochs):

        print("##Epoch", e, batch_size)
        testing_counter = 0
        steps_per_epoch = int(x_train.shape[0] / base_batch_size)
        print("Steps per epoch:", steps_per_epoch)
        for i in range(steps_per_epoch):

            start = time.time()

            images, labels = next(data_gen)

            if warmup_iter > 0:

                assert len(labels) == batch_size
                model_fit(images, labels, iteration)
                sample_count += len(labels)
                iteration += 1
                warmup_iter -= 1
                continue

            elif mode == BASELINE:

                # Note: this mode calls fit only once

                assert len(labels) == batch_size
                model_fit(images, labels, iteration)
                sample_count += len(labels)
                iteration += 1

            elif mode == ITER_MODE:

                # iterate multiple times on one large batch
                # iteratively picking the most difficult samples
                # and using only these for training

                compute_losses_once = params['compute_losses_once']
                if compute_losses_once:
                    losses = compute_losses(images, labels)

                for iter in range(params['steps']):

                    #print("Iteration", iter)
                    if not compute_losses_once:
                        losses = compute_losses(images, labels)

                    retry_images, retry_labels, _ = sample_by_loss(images, labels, base_batch_size, losses)

                    assert len(retry_labels) == base_batch_size
                    model_fit(retry_images, retry_labels, iteration)
                    sample_count += len(retry_labels)
                    iteration += 1

            elif mode == ALL_POOL_MODE:

                # finds the worst samples from the whole dataset
                # i.e. use the whole dataset as pool
                # (yes, this is going to be slow...)
                # This model doe not use the current batch in any way

                extra_sampling = params.get('extra_sampling', 0)
                fitted = fit_all_pool(batch_size, params, iteration, extra_sampling=extra_sampling)
                sample_count += fitted
                iteration += 1

                fitted = fit_all_pool(k, params, iteration)
                sample_count += fitted
                iteration += 1
            elif mode == CURIOSITY_ALL_POOL_MODE:

                # fit current batch, than fit the hardest samples
                # from the whole dataset

                # here no extra_sampling means to use current batch
                extra_sampling = params.get('extra_sampling', -1)
                if extra_sampling == -1:
                    assert len(labels) == batch_size
                    model_fit(images, labels, iteration)
                    sample_count += len(labels)
                    iteration += 1
                else:
                    #if random(0, 10) == 0:
                    #    fitted = fit_all_pool(batch_size, params, iteration, extra_sampling=extra_sampling, easiest=True)
                    #else:
                    #    fitted = fit_all_pool(batch_size, params, iteration, extra_sampling=extra_sampling)
                    fitted = fit_all_pool(batch_size, params, iteration, extra_sampling=extra_sampling)
                    sample_count += fitted
                    iteration += 1

                # difficult batch
                fitted = fit_all_pool(k, params, iteration)
                sample_count += fitted
                iteration += 1

            elif mode == POOL_MODE:

                # does extra training with random samples
                # from the pool of most difficult samples

                # it seems like the best BASE_POOL_SIZE depends on the cr value
                BASE_POOL_SIZE=params['pool_size']
                # I use this as an optimization only, not to recompute the losses
                # on each iteration.
                MAX_POOL_SIZE=BASE_POOL_SIZE * params['pool_max_size_factor']

                if pool_images is None:
                    print("Init pool")
                    indexes = np.arange(x_train.shape[0])
                    # add a few samples to have something to start with
                    retry_idx = np.random.choice(indexes, size=batch_size, replace=False)
                    pool_images = x_train[retry_idx]
                    pool_labels = y_train[retry_idx]

                # train with a fresh batch
                assert len(labels) == batch_size
                model_fit(images, labels, iteration)
                sample_count += len(labels)
                iteration += 1

                # append hard samples to the pool
                full_sample = False # it seems to be worse (?)
                if full_sample:
                    # compute this just to compute the pool ratio later
                    losses = compute_losses(images, labels)

                    indexes = np.arange(x_train.shape[0])
                    retry_idx = np.random.choice(indexes, size=k, replace=False)
                    new_pool_images = x_train[retry_idx]
                    new_pool_labels = y_train[retry_idx]
                else:
                    losses = compute_losses(images, labels)
                    new_pool_images, new_pool_labels, new_pool_losses = sample_by_loss(images, labels, k, losses)

                    print(np.mean(new_pool_losses), np.mean(losses))

                    assert (np.mean(new_pool_losses) >= np.mean(losses)
                            or np.isclose(np.mean(new_pool_losses), np.mean(losses)))

                pool_images = np.append(pool_images, new_pool_images, axis=0)
                pool_labels = np.append(pool_labels, new_pool_labels, axis=0)

                # pick the samples for the extra training
                indexes = np.arange(pool_images.shape[0])
                #   note: this sampling could depend on the losses using p kwarg
                retry_idx = np.random.choice(indexes, size=k, replace=False)
                retry_images = pool_images[retry_idx]
                retry_labels = pool_labels[retry_idx]

                # train with hard samples
                assert len(retry_labels) == k
                model_fit(retry_images, retry_labels, iteration)
                sample_count += len(retry_labels)
                iteration += 1

                # if pool is too large, drop
                if pool_images.shape[0] > MAX_POOL_SIZE:
                    print("Reduce pool", pool_images.shape[0])
                    pool_losses = compute_losses(pool_images, pool_labels)
                    pre_mean_loss = np.mean(pool_losses)
                    #print("Mean losses", pre_mean_loss)
                    #print(pool_losses)
                    sort_idx = np.argsort(-pool_losses, axis=0) # reversed
                    #print(sort_idx)
                    sort_idx = sort_idx[:BASE_POOL_SIZE]
                    # copy to disconnect from the old array
                    pool_images = pool_images[sort_idx].copy()
                    pool_labels = pool_labels[sort_idx].copy()
                    pool_losses = pool_losses[sort_idx].copy()
                    #print("Post", np.mean(pool_losses))
                    assert np.mean(pool_losses) >= pre_mean_loss
                    #print(pool_losses)
                    print("Reduce pool done", pool_images.shape[0],
                          "pool loss ratio:",  np.mean(pool_losses)/np.mean(losses),
                          "- avg pool/train: ", np.mean(pool_losses), "/", np.mean(losses))
                    #exit(1)

            elif mode == WEIGHTS_MODE:

                # use keras sample weights to make the most difficult
                # samples more important

                first_images = images[:base_batch_size]
                first_labels = labels[:base_batch_size]

                assert len(first_labels) == base_batch_size
                fitted = fit_weights_mode(first_images, first_labels, params['scale_max'], iteration)
                sample_count += fitted
                iteration += 1

                second_images = images[base_batch_size:]
                second_labels = labels[base_batch_size:]

                assert len(second_labels) == k
                fitted = fit_weights_mode(second_images, second_labels, params['scale_max'], iteration)
                sample_count += fitted
                iteration += 1

            elif mode == CURIOSITY_MODE_SINGLE_BATCH:

                # does extra training with the most difficult
                # samples from the batch calling fit just once

                # Note: this mode calls fit only once
                losses = compute_losses(images, labels)

                retry_images, retry_labels, _ = sample_by_loss(images, labels, k, losses)

                joined_images = np.append(images, retry_images, axis=0)
                joined_labels = np.append(labels, retry_labels, axis=0)

                assert len(joined_labels) == batch_size+k
                model_fit(joined_images, joined_labels, iteration)
                sample_count += len(joined_labels)
                iteration += 1

            elif mode == CURIOSITY_MODE:

                # does extra training with the most difficult
                # samples from the batch

                assert len(labels) == batch_size
                model_fit(images, labels, iteration)
                sample_count += len(labels)
                iteration += 1

                # computing losses after fit makes more sense
                # and seems to work better
                losses = compute_losses(images, labels)

                retry_images, retry_labels, _ = sample_by_loss(images, labels, k, losses)

                assert len(retry_labels) == k
                model.fit(retry_images, retry_labels,
                          batch_size=k,
                          epochs=iteration+1,
                          initial_epoch=iteration,
                          verbose=0,
                          shuffle=False,
                          callbacks=train_callbacks)
                sample_count += len(retry_labels)
                iteration += 1

            elif mode == CURIOSITY_BASELINE:

                # does extra training with randomly selected samples
                # from the same batch
                assert len(labels) == batch_size
                model_fit(images, labels, iteration)
                sample_count += len(labels)
                iteration += 1

                # sample over the same batch
                indexes = np.arange(batch_size)
                retry_idx = np.random.choice(indexes, size=k, replace=False)
                retry_images = images[retry_idx]
                retry_labels = labels[retry_idx]

                assert len(retry_labels) == k
                model_fit(retry_images, retry_labels, iteration)
                sample_count += len(retry_labels)
                iteration += 1

            elif mode == CURIOSITY_BASELINE_FULL_SAMPLE:

                # does extra training with randomly selected samples
                # from the whole dataset
                assert len(labels) == batch_size
                model_fit(images, labels, iteration)
                sample_count += len(labels)
                iteration += 1

                # sample over the whole training set
                indexes = np.arange(x_train.shape[0])
                retry_idx = np.random.choice(indexes, size=k, replace=False)
                retry_images = x_train[retry_idx]
                retry_labels = y_train[retry_idx]

                assert len(retry_labels) == k
                model_fit(retry_images, retry_labels, iteration)
                sample_count += len(retry_labels)
                iteration += 1

            else:
                raise Exception("Unsupported mode " + str(mode))

            real_epochs = sample_count / x_train.shape[0]  # number of iterations over the whole dataset
            print("Processed samples", sample_count, "elapsed:", time.time() - start, f"(Real epochs: {round(real_epochs, 2)})")

            testing_counter += base_batch_size
            #if testing_counter > record_steps:
            if i % VALIDATION_ITERATIONS == 0:
                testing_counter = 0
                print(f"Testing model... (iter_with_no_improvements: {iter_with_no_improvements}")

                curr_loss, curr_acc = model.evaluate(x_test, y_test, callbacks=eval_callbacks)
                print("loss_acc", curr_loss, curr_acc)

                smooth_size = 15 #25
                smoothed_acc = np.mean(validation[-smooth_size:]) if len(validation) > smooth_size else np.mean(validation)
                print("smoothed_acc", smoothed_acc, "curr_acc", curr_acc)
                smoothed_acc_diff = (curr_acc - smoothed_acc) / smoothed_acc
                print("smoothed_difference", smoothed_acc_diff * 100, "%", "smoothed_acc", smoothed_acc, "curr_acc", curr_acc)

                for name, value in [("loss", curr_loss), ("accuracy", curr_acc),
                                    ("smoothed_acc", smoothed_acc), ("smoothed_acc_diff", smoothed_acc_diff)]:
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value.item()
                    summary_value.tag = name
                    val_writer.add_summary(summary, iteration)  # or sample count?
                val_writer.flush()

                validation.append(curr_acc)
                validation_by_sample_count.append((sample_count, curr_loss, curr_acc))

                if curr_acc > absolute_best_acc:
                    absolute_best_acc = curr_acc

                # early stop check
                use_smoothed_acc=True
                if use_smoothed_acc:
                    early_stop_curr_acc = smoothed_acc
                else:
                    early_stop_curr_acc = curr_acc

                perc_difference = (early_stop_curr_acc - best_acc) / best_acc
                print("perc_difference", perc_difference * 100, "%", "best", best_acc, "curr", early_stop_curr_acc)
                if perc_difference <= EARLY_STOP_TOLERANCE:
                    iter_with_no_improvements += 1
                else:
                    iter_with_no_improvements = 0
                    best_acc = early_stop_curr_acc
                    best_acc_iter = iteration

                print("iter_with_no_improvements", iter_with_no_improvements)

                if iter_with_no_improvements > EARLY_STOP_PATIENCE:
                    best_accuracies.append((best_acc, best_acc_iter, time.time() - train_start))
                    break

        if iter_with_no_improvements > EARLY_STOP_PATIENCE:
            break

    real_epochs = sample_count/x_train.shape[0]     # number of iterations over the whole dataset
    fprint("Total processed samples", sample_count,  "elapsed:", time.time() - train_start)
    fprint("Best accuracy", best_acc, "at iteration", best_acc_iter, f"(Real epochs: {round(real_epochs, 2)})")
    fprint(f"Absolute best: {absolute_best_acc}")
    return validation, validation_by_sample_count, best_accuracies


def model_fit(images, labels, k_epoch):
    batch_size = len(labels)
    model.fit(images, labels,
              batch_size=batch_size,
              epochs=k_epoch + 1,
              verbose=0,
              initial_epoch=k_epoch,
              shuffle=False,
              callbacks=train_callbacks)


# https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
def sample_by_loss(images, labels, size, losses, extra_sampling=DEFAULT_EXTRA_SAMPLING, easiest=False):

    # Use "extra_sampling" to get a "fuzzy" sampling of the most difficult items

    assert extra_sampling >= 0

    if extra_sampling == 0:
        if not easiest:
            worst = np.argpartition(losses, -size)
            retry_idx = worst[-size:]
            retry_images = images[retry_idx]
            retry_labels = labels[retry_idx]
            return retry_images, retry_labels, losses[retry_idx]
        else:
            best = np.argpartition(losses, size)
            retry_idx = best[:size]
            retry_images = images[retry_idx]
            retry_labels = labels[retry_idx]
            return retry_images, retry_labels, losses[retry_idx]
    else:
        if easiest:
            raise Exception("Easy not supported with extra_sampling")

        ext_size = int(size * (1 + extra_sampling))
        worst = np.argpartition(losses, -ext_size)
        retry_idx = worst[-ext_size:]
        sub_images = images[retry_idx].copy()
        sub_labels = labels[retry_idx].copy()
        sub_losses = losses[retry_idx].copy()

        indexes = np.arange(ext_size)
        sub_choice_idx = np.random.choice(indexes, size=size, replace=False)

        retry_images = sub_images[sub_choice_idx]
        retry_labels = sub_labels[sub_choice_idx]
        return retry_images, retry_labels, sub_losses[sub_choice_idx]


def fit_all_pool(batch_size, params, k_epoch, extra_sampling=0, easiest=False):

    pool_size = int(x_train.shape[0] * params['dataset_ratio'])
    #print("fit_all_pool", batch_size, pool_size)

    indexes = np.arange(x_train.shape[0])
    pool_idx = np.random.choice(indexes, size=pool_size, replace=False)
    pool_images = x_train[pool_idx].copy()
    pool_labels = y_train[pool_idx].copy()

    losses = compute_losses(pool_images, pool_labels)

    retry_images, retry_labels, _ = sample_by_loss(pool_images, pool_labels, batch_size, losses, extra_sampling=extra_sampling, easiest=easiest)

    assert len(retry_labels) == batch_size
    model_fit(retry_images, retry_labels, k_epoch)

    return batch_size


def fit_weights_mode(images, labels, scale_max, k_epoch):

    losses = compute_losses(images, labels)

    # min/max 1 to 10
    proc_losses = scale(losses, min(losses), max(losses), 1, scale_max)

    # proc_losses = np.square(proc_losses)

    # proc_losses = softmax(proc_losses)

    # proc_losses = translate(proc_losses, min(proc_losses), max(proc_losses), 1, 100)

    # norm
    # proc_losses /= np.sum(proc_losses)

    # try to make it comparable to the other weights set
    # proc_losses *= batch_size

    # temp
    # sample_weights = np.ones(shape=(batch_size)) * fix

    sample_weights = proc_losses

    batch_size = len(labels)

    # Here I use the sample_weights (do not replace this with model_fit)
    model.fit(images, labels,
              batch_size=batch_size,
              epochs=k_epoch+1,
              initial_epoch=k_epoch,
              verbose=0,
              sample_weight=sample_weights,
              shuffle=False,
              callbacks=train_callbacks)

    return len(labels)


loss_func = None
def compute_losses(images, labels):

    global loss_func

    if loss_func is None:
        y_true = Input(shape=(10,))
        ce = K.categorical_crossentropy(y_true, model.output)
        loss_func = K.function(model.inputs + [y_true], [ce])

    losses = loss_func([images, labels])[0]
    return losses

#runs = [(CURIOSITY_MODE, 0.25, {}), (CURIOSITY_BASELINE, 0.25, {})]

all_runs = [
        (BASELINE, 1, {}),
        (CURIOSITY_MODE, 1, {}),
        (CURIOSITY_BASELINE, 1, {}),
        (CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}),
        (CURIOSITY_MODE_SINGLE_BATCH, 1, {}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.02}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 10, 'compute_losses_once': False}),
        (WEIGHTS_MODE, 1, {'scale_max': 100})]


runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 10, 'compute_losses_once': False}),
        (WEIGHTS_MODE, 1, {'scale_max': 100})]

#runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {})]

#runs = [(ALL_POOL_MODE, 1, {'dataset_ratio': 0.1}),
#        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5})]

runs = [
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.002}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.005}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.01}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.02}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.03}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.05})]

runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.0}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 10, 'compute_losses_once': False}),
        (WEIGHTS_MODE, 1, {'scale_max': 100})]

runs = [(ITER_MODE, 1, {'ratio': 4, 'steps': 4, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 4, 'steps': 10, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 4, 'steps': 20, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 4, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 10, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 20, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 20, 'steps': 4, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 20, 'steps': 10, 'compute_losses_once': False}),
        (ITER_MODE, 1, {'ratio': 20, 'steps': 20, 'compute_losses_once': False}),
        ]

runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5, 'warmup_iter': 10}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5, 'warmup_iter': 100}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5, 'warmup_iter': 250}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5, 'warmup_iter': 500})]

runs = [(CURIOSITY_BASELINE, 0.25, {}),
        (CURIOSITY_BASELINE_FULL_SAMPLE, 0.25, {}),
        (CURIOSITY_MODE, 0.25, {})]

runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}),
        (POOL_MODE, 1, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5}),
        (ALL_POOL_MODE, 1, {'dataset_ratio': 0.01}),
        (ITER_MODE, 1, {'ratio': 10, 'steps': 10, 'compute_losses_once': False}),
        (WEIGHTS_MODE, 1, {'scale_max': 100})]

runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 0.25, {}),
        (CURIOSITY_MODE, 0.25, {}),
        (POOL_MODE, 0.25, {'pool_size': 100 * 10, 'pool_max_size_factor': 1.5}),
        (CURIOSITY_ALL_POOL_MODE, 0.25, {'dataset_ratio': 0.02})]

runs = [#(CURIOSITY_BASELINE_FULL_SAMPLE, 0.25, {}),
        #(CURIOSITY_ALL_POOL_MODE, 0.25, {'dataset_ratio': 0.02}),
        (CURIOSITY_ALL_POOL_MODE, 0.25, {'dataset_ratio': 0.02, 'extra_sampling': 8})]


#runs = [(ALL_POOL_MODE, 1, {'dataset_ratio': 0.02})]

#runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {})]

#runs = [(CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}), (WEIGHTS_MODE, 1, {'scale_max': 100})]

#runs = [(BASELINE, 1, {}), (CURIOSITY_BASELINE, 1, {}), (CURIOSITY_BASELINE_FULL_SAMPLE, 1, {}),
#        (CURIOSITY_MODE, 1, {}), (CURIOSITY_MODE_SINGLE_BATCH, 1, {})]

#runs = [(CURIOSITY_MODE, 0.1, {}), (CURIOSITY_MODE, 0.2, {}), (CURIOSITY_MODE, 0.3, {}), (CURIOSITY_MODE, 0.4, {}),
#        (CURIOSITY_MODE, 0.5, {}), (CURIOSITY_MODE, 0.6, {}), (CURIOSITY_MODE, 0.7, {}), (CURIOSITY_MODE, 0.8, {}),
#        (CURIOSITY_MODE, 0.9, {}), (CURIOSITY_MODE, 1, {}),]


base_batch_size = 100

notes = sys.argv[2]

shutil.copy(sys.argv[0], "data/" + name)

fprint(f"NAME: {name}")
fprint(f"NOTES: {notes}")
fprint(f"PARAMS: SEED {SEED}, DEFAULT_EXTRA_SAMPLING {DEFAULT_EXTRA_SAMPLING}, shuffle {shuffle}")
fprint(f"PARAMS: EARLY_STOP_PATIENCE {EARLY_STOP_PATIENCE}, EARLY_STOP_TOLERANCE {EARLY_STOP_TOLERANCE}")
fprint(f"PARAMS: epochs {epochs}, average_count {average_count}, base_batch_size {base_batch_size}")
fprint("RUNS:", runs)

for i, run in enumerate(runs):

    # all runs use the dataset in the same order (where applicable)
    set_seed(SEED)

    avg_validations = []
    avg_validations_by_sample_count = []

    run_name = f"run_{i}_{run[0]}"

    for ac in range(average_count):

        model = None
        loss_func = None
        K.clear_session()

        model = create_model()

        logdir = f"logs/{name}/{name}_{run_name}_avg_{ac}"
        val_writer = tf.summary.FileWriter(logdir)

        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        #eval_callbacks = [tensorboard_callback]
        eval_callbacks = []

        start = time.time()
        fprint(f"{i} RUN {run} avg_iter: {ac} started.")
        fprint(f"Train params {run[2]}")
        validation, validation_by_sample_count, best_accuracies = train(run[0], base_batch_size,
                                                       curiosity_ratio=run[1], params=run[2])
        fprint(f"{i} RUN {run} avg_iter: {ac} done. Elapsed: ", time.time() - start)

        avg_validations.append(validation)
        np.savetxt(f'{root}/data_{name}_{run_name}_{ac}', validation, delimiter=',')

        avg_validations_by_sample_count.append(validation_by_sample_count)
        np.savetxt(f'{root}/data_{name}_{run_name}_{ac}_by_sample_count', validation_by_sample_count, delimiter=',')

        val_writer.close()

    min_len = min([len(x) for x in avg_validations])
    cut_avg_validations = [x[:min_len] for x in avg_validations]
    avg_validation = np.mean(cut_avg_validations, axis=0)
    np.savetxt(f'{root}/data_{name}_{run_name}_avg', avg_validation, delimiter=',')

    print("avg len", [len(x) for x in avg_validations_by_sample_count])
    min_len = min([len(x) for x in avg_validations_by_sample_count])
    cut_avg_validations_by_sample_count = [x[:min_len] for x in avg_validations_by_sample_count]
    avg_validation_by_sample_count = np.mean(cut_avg_validations_by_sample_count, axis=0)
    np.savetxt(f'{root}/data_{name}_{run_name}_avg_by_sample_count', avg_validation_by_sample_count, delimiter=',')

    # add accuracy on tensorboard
    logdir = f"logs/{name}/{name}_{run_name}_avg"
    avg_val_writer = tf.summary.FileWriter(logdir)
    for iter, loss, acc in avg_validation_by_sample_count:
        summary = tf.Summary(value=[tf.Summary.Value(tag="average_acc",
                                                     simple_value=acc)])
        avg_val_writer.add_summary(summary, iter)
    avg_val_writer.flush()
    avg_val_writer.close()

    best_accuracies = np.asarray(best_accuracies)
    avg_acc = np.mean(best_accuracies[:, 0])
    avg_epoch = np.mean(best_accuracies[:, 1])
    avg_elapsed = np.mean(best_accuracies[:, 2])
    fprint(f"Best average accuracy for run {run_name}: {avg_acc} at average epoch {avg_epoch} (avg_elapsed {avg_elapsed})")


