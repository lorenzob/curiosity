
# based on: https://keras.io/examples/mnist_cnn/

"""

Here is a brief summary of the different training modes.

Right now the most interesting one is CURIOSITY_FULL_MODE. All these mode are directly comparable (when all run with
single_batch or two batches).


BASELINE
Classic training, process the dataset sequentially.

CURIOSITY_MODE
This is the one described in the article. Pick a batch, train on it, compute losses, retrain on the most difficult
subset of this same batch. This second batch is called "extra batch".
This mode, like many others, can be run with "single_batch" or not (see below for the rationale for this).

CURIOSITY_BASELINE
Just like CURIOSITY_MODE but the extra batch is randomly selected from the first batch. This mode exists only as a
strict baseline for CURIOSITY_MODE. It does not make sense to use it in any others way. This mode should perform worse
than BASELINE. If CURIOSITY_MODE performs better than BASELINE it means it also overcomes for this disadvantage.

CURIOSITY_POOL_MODE
It keeps a pool of the most difficult samples encountered in each batch. Like CURIOSITY_MODE but extra batch is
randomly sampled from the pool.

CURIOSITY_FULL_MODE
Like CURIOSITY_MODE but the extra batch is randomly sampled from the hardest samples of the whole dataset.

WEIGHTS_MODE
It uses the samples weights. It picks a "normal" batch, compute the losses, and map the losses to samples
weights and fits the batch with these params.

NOTE: All modes fits the same number of items (batch_size+k) per iteration. Most modes can do this in a single fit
call or with two fit calls.

# Notes

I think there are several sources of difference in the training (accuracy and speed) for a fixed dataset:
	- amount of samples fitted
	- amount of different samples fitted
	- amount of backprop passes
	- batch size
	- samples weights
	- difficulty of the samples (?)

What I'm trying to isolate with the curiosity modes is the contribution of the last one.

Dataset order is the same for each run, but differs for each average iteration.


# Why "single_batch"
Calling fit once with a large batch or twice with two "half" batches is obviously different as two calls mean two
pass of back propagation. Some modes are naturally "single mode", CLASSIC, others are two steps modes. Having only two
steps modes might be the most natural choice but this make dangerous to compare different curiosity ratios: calling fit
twice with two batches of size 10 and 90 might be different from two batches with sizes 50 and 50 (especially if the
smaller batch is a difficult one).

Also, MAYBE, there is a slightly difference between one step and two steps. With on step the normal samples and the hard
ones are "averaged" together during the backprop step, with two separate calls the hardest samples alone might have
a strongest impact(?). I have no (reasonable) idea on how to isolate this difference.

So two steps is the most natural option but "single_batch" makes it easier to make some kind of comparisons. It also
allows to use the same baseline for different tests.

Note: there can be subtle differences between the same mode run in single or in "double" mode, for example with
CURIOSITY_MODE the first call to fit alters the loss so the choice of the extra batch changes slightly.



"""

from keras import Sequential, Model, regularizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, MaxPool2D

import os
import random
import shutil
import sys
import time

import keras
from keras.datasets import mnist, cifar10
from keras import backend as K, Input

import numpy as np

import tensorflow as tf

num_classes = 10
# default: 3, 2
# big comparison: 6, 3
epochs = 100
average_count = 3

VALIDATION_ITERATIONS = 100

EARLY_STOP_PATIENCE = 15
EARLY_STOP_TOLERANCE = 0.1 / 100

# debug settings
quick_debug= False
if quick_debug:
    epochs = 2
    average_count = 2
    EARLY_STOP_PATIENCE = 2
    EARLY_STOP_TOLERANCE = 1 / 100


SEED=123

DEFAULT_SOFT_SAMPLING=0

shuffle = True

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)

name = sys.argv[1]

root=f"logs/{name}"
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


# default input image dimensions
img_rows, img_cols, channels = 28, 28, 1

# the data, split between train and test sets
#dataset_name = 'CIFAR'
dataset_name = 'MNIST'
#dataset_name = 'Fashion'
fprint("Dataset ", dataset_name)
if dataset_name == 'MNIST':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
elif dataset_name == 'Fashion':
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif dataset_name == 'CIFAR':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    img_rows, img_cols, channels = 32, 32, 3

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
    input_shape = (channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

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


def create_simple_model():

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

    return model

def create_MobileNetV2_model():

    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape,include_top=False,
                                                             weights='imagenet',
                                                             classes=10)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=base_model.input, output=predictions)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    return model


def create_Adv2_model():

    # https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model

#https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
def create_adv_model():

    # also try:
    # https://www.kaggle.com/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1

    fprint("Advanced model")

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu', input_shape=input_shape))
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

    return model


def create_model():

    learning_rate=0.001
    fprint("learning_rate", learning_rate)

    model_name = "SIMPLE"
    fprint("model_name", model_name)
    if model_name == "SIMPLE":
        model = create_simple_model()
    elif model_name == "ADV":
        model = create_adv_model()
    elif model_name == "ADV2":
        model = create_Adv2_model()
    elif model_name == "MOBILENET":
        model = create_MobileNetV2_model()
        learning_rate = 0.0001

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
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
CURIOSITY_BASELINE = 'CURIOSITY_BASELINE'

CURIOSITY_MODE = 'CURIOSITY_MODE'
CURIOSITY_POOL_MODE = 'CURIOSITY_POOL_MODE'
CURIOSITY_FULL_MODE = 'CURIOSITY_FULL_MODE'

WEIGHTS_MODE = 'WEIGHTS_MODE'  # keras sample weights based on loss

# please ignore these, they are here only for sanity checks
SWITCH_MODE = 'SWITCH_MODE'
FULL_MODE='FULL_MODE'
CURIOSITY_CONSUME_MODE='CURIOSITY_CONSUME_MODE'

def train(mode, base_batch_size, curiosity_ratio=1, params=None):

    print(f"Mode {mode}, curiosity_ratio: {curiosity_ratio}")

    pool_images = None
    pool_labels = None

    k = int(base_batch_size * curiosity_ratio)

    if mode in [BASELINE, WEIGHTS_MODE, FULL_MODE, SWITCH_MODE]:
        batch_size = base_batch_size
    else:
        batch_size = base_batch_size - k

    data_gen = data_generator_mnist(x_train, y_train, batch_size)

    warmup_iter = params.get('warmup_iter', 0)
    print("warmup_iter", warmup_iter)

    validation = []
    validation_by_iteration = []
    best_accuracies = []
    sample_count = 0
    train_start = time.time()
    iteration = 0   # number of fit call (note: some iterations process smaller batches than others)
    iter_with_no_improvements = 0
    best_loss = 1000    # very large numeber
    best_loss_iter = 0
    absolute_best_acc = 0
    absolute_best_acc_iter = 0
    original_mode = mode

    single_batch = params.get('single_batch', False)
    fprint("single_batch", single_batch)

    for e in range(epochs):

        print("##Epoch", e, batch_size)
        steps_per_epoch = int(x_train.shape[0] / base_batch_size)
        print("Steps per epoch:", steps_per_epoch)
        for i in range(steps_per_epoch):

            start = time.time()

            images, labels = next(data_gen)

            if original_mode == SWITCH_MODE:
                if mode == SWITCH_MODE:
                    mode = FULL_MODE    # it switches immediately
                if i % params['switch_count'] == 0:
                    mode = BASELINE if mode == FULL_MODE else FULL_MODE
                    print("Switching mode to", mode)

            if warmup_iter > 0 and iteration < warmup_iter:

                print("Warmup", iteration, '/', warmup_iter, " - Next mode", mode)

                if single_batch:
                    #assert len(labels) == base_batch_size
                    #fitted = model_fit(images, labels, iteration)
                    #sample_count += fitted
                    #iteration += 1
                    raise Exception("Not supported")
                else:
                    # works only with cr 50
                    fitted = model_fit(images, labels, iteration)
                    sample_count += fitted
                    iteration += 1

                    images, labels = next(data_gen)
                    fitted = model_fit(images, labels, iteration)
                    sample_count += fitted
                    iteration += 1

            elif mode == BASELINE:

                if single_batch:
                    assert len(labels) == batch_size
                    fitted = model_fit(images, labels, iteration)
                    sample_count += fitted
                    iteration += 1
                else:
                    fitted = fit_batch_in_two_steps(images, labels, batch_size-k, k, iteration)
                    sample_count += fitted
                    iteration += 2

            elif mode == CURIOSITY_MODE:

                # does an extra training step with the most difficult
                # samples from the batch

                if single_batch:
                    losses = compute_losses(images, labels)

                    retry_images, retry_labels, _ = sample_by_loss(images, labels, k, losses)

                    joined_images = np.append(images, retry_images, axis=0)
                    joined_labels = np.append(labels, retry_labels, axis=0)

                    assert len(joined_labels) == batch_size + k
                    fitted = model_fit(joined_images, joined_labels, iteration)
                    sample_count += fitted
                    iteration += 1

                else:

                    losses_first = params.get('losses_first', False)
                    if losses_first:
                        losses = compute_losses(images, labels)

                    assert len(labels) == batch_size
                    fitted = model_fit(images, labels, iteration)
                    sample_count += fitted
                    iteration += 1

                    # computing losses after fit makes more sense
                    # and seems to work better
                    if not losses_first:
                        losses = compute_losses(images, labels)
                    retry_images, retry_labels, _ = sample_by_loss(images, labels, k, losses)

                    assert len(retry_labels) == k
                    model_fit(retry_images, retry_labels, iteration)
                    sample_count += len(retry_labels)
                    iteration += 1

            elif mode == CURIOSITY_BASELINE:

                # does extra training with randomly selected samples
                # from the same batch
                if single_batch:

                    indexes = np.arange(batch_size)
                    retry_idx = np.random.choice(indexes, size=k, replace=False)
                    retry_images = images[retry_idx]
                    retry_labels = labels[retry_idx]

                    joined_images = np.append(images, retry_images, axis=0)
                    joined_labels = np.append(labels, retry_labels, axis=0)

                    assert len(joined_labels) == batch_size + k
                    fitted = model_fit(joined_images, joined_labels, iteration)
                    sample_count += fitted
                    iteration += 1

                else:

                    assert len(labels) == batch_size
                    fitted = model_fit(images, labels, iteration)
                    sample_count += fitted
                    iteration += 1

                    # retrain with random samples
                    indexes = np.arange(batch_size)
                    retry_idx = np.random.choice(indexes, size=k, replace=False)
                    retry_images = images[retry_idx]
                    retry_labels = labels[retry_idx]

                    assert len(retry_labels) == k
                    fitted = model_fit(retry_images, retry_labels, iteration)
                    sample_count += fitted
                    iteration += 1

            elif mode == CURIOSITY_CONSUME_MODE:

                if single_batch:
                    raise Exception("Not supported")
                else:

                    assert len(labels) == batch_size
                    fitted = model_fit(images, labels, iteration)
                    sample_count += fitted
                    iteration += 1

                    if pool_labels is None or len(pool_labels) < k:

                        print("Filling pool, curr size: ", len(pool_labels) if pool_labels is not None else None)

                        pool_size = params['pool_size']
                        sub_images, sub_labels = dataset_random_subset(pool_size * 5)
                        sub_losses = compute_losses(sub_images, sub_labels)

                        #pool_soft_sampling = params.get('pool_soft_sampling', 0)
                        pool_images, pool_labels, losses = sample_by_loss(sub_images, sub_labels, pool_size, sub_losses)

                        print("New pool size: ", len(pool_labels))

                    len_pre_sample = len(pool_labels)

                    indexes = np.arange(pool_labels.shape[0])
                    retry_idx = np.random.choice(indexes, size=k, replace=False)

                    retry_images = pool_images[retry_idx]
                    retry_labels = pool_labels[retry_idx]

                    # consume the used images
                    del_mask = np.ones(len(pool_labels), dtype=bool)
                    del_mask[retry_idx] = False
                    pool_images = pool_images[del_mask].copy()
                    pool_labels = pool_labels[del_mask].copy()

                    print("Pool size after sample: ", len(pool_labels))

                    print(len(pool_labels) , (len_pre_sample - k))
                    assert len(pool_labels) == (len_pre_sample - k)

                    assert len(retry_labels) == k
                    fitted = model_fit(retry_images, retry_labels, iteration)
                    sample_count += fitted
                    iteration += 1

            elif mode == CURIOSITY_FULL_MODE:

                # does extra training with randomly selected difficult samples
                # from the whole dataset

                if single_batch:

                    retry_images, retry_labels = sample_by_loss_from_full_dataset(k, params)

                    joined_images = np.append(images, retry_images, axis=0)
                    joined_labels = np.append(labels, retry_labels, axis=0)

                    assert len(joined_labels) == batch_size + k
                    fitted = model_fit(joined_images, joined_labels, iteration)
                    sample_count += fitted
                    iteration += 1

                else:

                    # here no soft_sampling means to use current batch rather
                    # than do a random sampling (is closer to the original idea
                    # and it avoids the miniscule risk of going over the same
                    # elements too much)
                    soft_sampling = params.get('soft_sampling', -1)
                    if soft_sampling == -1:
                        assert len(labels) == batch_size
                        fitted = model_fit(images, labels, iteration)
                    else:
                        fitted = fit_by_loss_from_full_dataset(batch_size, params, iteration)
                    sample_count += fitted
                    iteration += 1

                    # difficult batch
                    fitted = fit_by_loss_from_full_dataset(k, params, iteration)
                    sample_count += fitted
                    iteration += 1

            elif mode == FULL_MODE:

                if single_batch:
                    raise Exception("Not supported")
                else:

                    hard_images, hard_labels = sample_by_loss_from_full_dataset(batch_size, params)

                    fitted = fit_batch_in_two_steps(hard_images, hard_labels, batch_size-k, k, iteration)
                    sample_count += fitted
                    iteration += 2

            elif mode == CURIOSITY_POOL_MODE:

                # does extra training with randomly selected samples
                # from the pool

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
                fitted = model_fit(images, labels, iteration)
                sample_count += fitted
                iteration += 1

                # append hard samples to the pool
                pick_new_elements_from_the_whole_dataset = False # it seems to be worse (?)
                if pick_new_elements_from_the_whole_dataset:
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
                fitted = model_fit(retry_images, retry_labels, iteration)
                sample_count += fitted
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

            else:
                raise Exception("Unknown mode " + str(mode))

            real_epochs = sample_count / x_train.shape[0]  # number of iterations over the whole dataset
            print("Iteration", iteration, ", processed samples", sample_count, "elapsed:", time.time() - start, f"(Real epochs: {round(real_epochs, 2)})")

            if i % VALIDATION_ITERATIONS == 0:
                print(f"Testing model... (iter_with_no_improvements: {iter_with_no_improvements}")

                # eval training data too to make the charts (only a subset to speed it up)
                indexes = np.arange(x_train.shape[0])
                pool_idx = np.random.choice(indexes, size=len(y_test), replace=False)
                x_train_sample = x_train[pool_idx].copy()
                y_train_sample = y_train[pool_idx].copy()
                train_loss, train_acc = model.evaluate(x_train_sample, y_train_sample)
                print("training loss_acc", train_loss, train_acc)

                for name, value in [("loss", train_loss), ("accuracy", train_acc)]:
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value.item()
                    summary_value.tag = name
                    train_writer.add_summary(summary, iteration)
                train_writer.flush()

                # eval test set
                test_loss, test_acc = model.evaluate(x_test, y_test, callbacks=eval_callbacks)
                print("test loss_acc", test_loss, test_acc)

                for name, value in [("loss", test_loss), ("accuracy", test_acc)]:
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    summary_value.simple_value = value.item()
                    summary_value.tag = name
                    test_writer.add_summary(summary, iteration)
                test_writer.flush()

                validation.append(test_acc)
                validation_by_iteration.append((iteration, test_loss, test_acc))

                if test_acc > absolute_best_acc:
                    absolute_best_acc = test_acc
                    absolute_best_acc_iter = iteration

                # early stop check
                early_stop_curr_loss = test_loss

                perc_difference = (early_stop_curr_loss - best_loss) / best_loss
                print("testing loss perc_difference", perc_difference * 100, "%", "prev best_loss", best_loss, "curr loss", early_stop_curr_loss)
                if perc_difference >= EARLY_STOP_TOLERANCE:
                    iter_with_no_improvements += 1
                else:
                    iter_with_no_improvements = 0
                    best_loss = early_stop_curr_loss
                    best_loss_iter = iteration

                print("iter_with_no_improvements", iter_with_no_improvements)

                if iter_with_no_improvements > EARLY_STOP_PATIENCE:
                    best_accuracies.append((absolute_best_acc, absolute_best_acc_iter, time.time() - train_start))
                    break

        if iter_with_no_improvements > EARLY_STOP_PATIENCE:
            break

    if len(best_accuracies) == 0:   # no early stopping
        best_accuracies.append((absolute_best_acc, absolute_best_acc_iter, time.time() - train_start))

    real_epochs = sample_count/x_train.shape[0]     # number of iterations over the whole dataset
    fprint("Total processed samples", sample_count, "elapsed:", time.time() - train_start)
    fprint("Best loss", best_loss, "at iteration", best_loss_iter, f"(Real epochs: {round(real_epochs, 2)})")
    fprint("Best accuracy", absolute_best_acc, "at iteration", absolute_best_acc_iter, f"(Real epochs: {round(real_epochs, 2)})")
    return validation, validation_by_iteration, best_accuracies


def fit_batch_in_two_steps(images, labels, batch_size, k, iteration):

    new_sample_count = 0

    first_images = images[:batch_size]
    first_labels = labels[:batch_size]
    assert len(first_labels) == batch_size
    fitted = model_fit(first_images, first_labels, iteration)
    new_sample_count += fitted

    second_images = images[batch_size:]
    second_labels = labels[batch_size:]
    assert len(second_labels) == k
    fitted = model_fit(second_images, second_labels, iteration)
    new_sample_count += fitted

    assert new_sample_count == (batch_size + k)
    return new_sample_count


def model_fit(images, labels, k_epoch):
    batch_size = len(labels)
    model.fit(images, labels,
              batch_size=batch_size,
              epochs=k_epoch + 1,
              verbose=1,
              initial_epoch=k_epoch,
              shuffle=False,
              callbacks=train_callbacks)
    return batch_size


# https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
def sample_by_loss(images, labels, size, losses, soft_sampling=DEFAULT_SOFT_SAMPLING, easiest=False):

    # Use "soft_sampling" to get a "fuzzy" sampling of the most difficult items

    assert soft_sampling >= 0

    if soft_sampling == 0:
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
            raise Exception("Easy not supported with soft_sampling")

        ext_size = int(size * (1 + soft_sampling))
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


def fit_by_loss_from_full_dataset(batch_size, params, k_epoch, easiest=False):

    retry_images, retry_labels = sample_by_loss_from_full_dataset(batch_size, params)

    assert len(retry_labels) == batch_size
    fitted = model_fit(retry_images, retry_labels, k_epoch)

    return fitted


def sample_by_loss_from_full_dataset(batch_size, params, easiest=False):

    if batch_size == 0:
        return [], []

    pool_size = int(x_train.shape[0] * params['dataset_ratio'])

    pool_images, pool_labels = dataset_random_subset(pool_size)

    losses = compute_losses(pool_images, pool_labels)

    soft_sampling = params.get('soft_sampling', 0)
    retry_images, retry_labels, _ = sample_by_loss(pool_images, pool_labels, batch_size, losses,
                                                   soft_sampling=soft_sampling, easiest=easiest)
    return retry_images, retry_labels


def dataset_random_subset(pool_size):

    indexes = np.arange(x_train.shape[0])
    pool_idx = np.random.choice(indexes, size=pool_size, replace=False)
    pool_images = x_train[pool_idx].copy()
    pool_labels = y_train[pool_idx].copy()
    return pool_images, pool_labels


def fit_weights_mode(images, labels, scale_max, k_epoch):

    losses = compute_losses(images, labels)

    # min/max 1 to 10
    proc_losses = scale(losses, min(losses), max(losses), 1, scale_max)

    # proc_losses = np.square(proc_losses)

    # proc_losses = softmax(proc_losses)

    # proc_losses = translate(proc_losses, min(proc_losses), max(proc_losses), 1, 100)

    # norm
    # proc_losses /= np.sum(proc_losses)

    # try to make it comparable to the others weights set
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
              verbose=1,
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

cur_comp_sb = [
        (CURIOSITY_FULL_MODE, 0.01, {'dataset_ratio': 0.02, 'name': 'CAP_FIX_01_sb', 'single_batch': True}),
        (CURIOSITY_FULL_MODE, 0.25, {'dataset_ratio': 0.02, 'name': 'CAP_FIX_25_sb', 'single_batch': True}),
        (CURIOSITY_FULL_MODE, 0.75, {'dataset_ratio': 0.02, 'name': 'CAP_FIX_75_sb', 'single_batch': True}),
        (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.02, 'name': 'CAP_FIX_50_sb', 'single_batch': True}),
        (CURIOSITY_FULL_MODE, 0.99, {'dataset_ratio': 0.02, 'name': 'CAP_FIX_99_sb', 'single_batch': True})]

fashion_comp = [(BASELINE, 0.25, {'dataset_ratio': 0.02, 'name': 'BL_FIX_25_2b', 'single_batch': False}),
                (CURIOSITY_FULL_MODE, 0.25, {'dataset_ratio': 0.02, 'name': 'CF_FIX_25_2b', 'single_batch': False}),
                (CURIOSITY_MODE, 0.25, {'name': 'CF_FIX_25_2b', 'single_batch': False})]

baseline_runs = [(BASELINE, 0.25, {'name': 'BL_FIX_25_sb', 'single_batch': True}),
                 (BASELINE, 0.5, {'name': 'BL_FIX_50_sb', 'single_batch': True}),
                 (CURIOSITY_BASELINE, 0.25, {'dataset_ratio': 0.02, 'name': 'CBL_FIX_25_sb', 'single_batch': True}),
                 (CURIOSITY_BASELINE, 0.5, {'dataset_ratio': 0.02, 'name': 'CBL_FIX_50_sb', 'single_batch': True})]

fashion_comp_ext = [(CURIOSITY_FULL_MODE, 0.1, {'dataset_ratio': 0.02, 'name': 'CF_FIX_10_2b', 'single_batch': False}),
                   (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.02, 'name': 'CF_FIX_50_2b', 'single_batch': False}),
                   (CURIOSITY_FULL_MODE, 0.25, {'dataset_ratio': 0.02, 'name': 'CF_FIX_25_2b_losses_first',
                                                'single_batch': False, 'losses_first': True})]

fashion_comp_ext2 = [(BASELINE, 0.5, {'name': 'BL_FIX_50_2b', 'single_batch': False}),
                     (CURIOSITY_MODE, 0.25, {'dataset_ratio': 0.02, 'name': 'CF_FIX_25_2b_losses_first',
                                                  'single_batch': False, 'losses_first': True})]

fashion_comp_sb = [(BASELINE, 0.5, {'dataset_ratio': 0.02, 'name': 'BL_FIX_50_sb', 'single_batch': True}),
                   (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.02, 'name': 'CF_FIX_50_sb', 'single_batch': True}),
                   (CURIOSITY_MODE, 0.5, {'name': 'CM_FIX_50_sb', 'single_batch': True})]

cifar_comp = [(BASELINE, 0.5, {'dataset_ratio': 0.02, 'name': 'BL_50_2b'}),
              (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.02, 'name': 'CF_50_2b'}),
              (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.01, 'name': 'CF_50_2b_001'}),
              (CURIOSITY_MODE, 0.3, {'name': 'CM_30_sb'})]

cifar_comp_ext = [(BASELINE, 0.5, {'name': 'BL_50_2b_bis'}),
              (CURIOSITY_FULL_MODE, 0.25, {'dataset_ratio': 0.01, 'name': 'CF_25_2b_001'}),
              (CURIOSITY_FULL_MODE, 0.75, {'dataset_ratio': 0.01, 'name': 'CF_75_2b_001'})]


#provare 4000/5000 o anche 2000
cifar_comp_wup = [(CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.01, 'name': 'CF_25_2b_wu05k', 'warmup_iter': 500}),
                  (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.01, 'name': 'CF_25_2b_wu1k', 'warmup_iter': 1000}),
                  (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.01, 'name': 'CF_25_2b_wu2k', 'warmup_iter': 2000}),
                  (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.01, 'name': 'CF_25_2b_wu4k', 'warmup_iter': 4000})]

# retry without warmup, 5 avg
cifar_comp_wup2 = [(BASELINE, 0.5, {'name': 'BL_50_2b_x2'}),
                  (CURIOSITY_FULL_MODE, 0.5, {'dataset_ratio': 0.01, 'name': 'CF_25_2b_wu2k_x2', 'warmup_iter': 2000})]

# todo
comp_switch = [(SWITCH_MODE, 0.5, {'name': 'SW_50_2b_sw5', 'dataset_ratio': 0.02, 'switch_count': 5}),
               (SWITCH_MODE, 0.5, {'name': 'SW_50_2b_sw10', 'dataset_ratio': 0.02, 'switch_count': 10}),
               (SWITCH_MODE, 0.5, {'name': 'SW_50_2b_sw25', 'dataset_ratio': 0.02, 'switch_count': 25}),
               (SWITCH_MODE, 0.5, {'name': 'SW_50_2b_sw50', 'dataset_ratio': 0.02, 'switch_count': 50}),
               (SWITCH_MODE, 0.5, {'name': 'SW_50_2b_sw100', 'dataset_ratio': 0.02, 'switch_count': 100})]

comp_switch = [(SWITCH_MODE, 0.25, {'name': 'SW_25_2b_sw25', 'dataset_ratio': 0.02, 'switch_count': 25}),
               (SWITCH_MODE, 0.5, {'name': 'SW_50_2b_sw25', 'dataset_ratio': 0.02, 'switch_count': 25}),
               (SWITCH_MODE, 0.75, {'name': 'SW_75_2b_sw25', 'dataset_ratio': 0.02, 'switch_count': 25})]

mnist_check_sb = [(BASELINE, 0.25, {'name': 'mc_BL_25_sb', 'single_batch': True}),
                  (CURIOSITY_FULL_MODE, 0.25, {'name': 'mc_CBL_FIX_25_sb', 'dataset_ratio': 0.02, 'single_batch': True}),
                  (BASELINE, 0.5, {'name': 'mc_BL_50_sb', 'single_batch': True}),
                  (CURIOSITY_FULL_MODE, 0.5, {'name': 'mc_CBL_FIX_50_sb', 'dataset_ratio': 0.02, 'single_batch': True})]

mnist_check_2b = [(BASELINE, 0.25, {'name': 'mnA_BL_25_2b', 'single_batch': False}),
                  (CURIOSITY_FULL_MODE, 0.25, {'name': 'mnA_CFM_25_2b', 'dataset_ratio': 0.02, 'single_batch': False}),
                  (BASELINE, 0.33, {'name': 'mnA_BL_33_2b', 'single_batch': False}),
                  (CURIOSITY_FULL_MODE, 0.33, {'name': 'mnA_CFM_33_2b', 'dataset_ratio': 0.02, 'single_batch': False}),
                  (BASELINE, 0.5, {'name': 'mnA_BL_50_2b', 'single_batch': False}),
                  (CURIOSITY_FULL_MODE, 0.5, {'name': 'mnA_CFM_50_2b', 'dataset_ratio': 0.02, 'single_batch': False})]

# with zero does not work well probably because is sees the same items over and over
mnist_full_2b = [(FULL_MODE, 0.5, {'name': 'mc_FM_50_2b', 'dataset_ratio': 0.045, 'single_batch': False, 'soft_sampling': 0}),
                 (FULL_MODE, 0.5, {'name': 'mc_FM_50_2b', 'dataset_ratio': 0.045, 'single_batch': False, 'soft_sampling': 2}),
                 (FULL_MODE, 0.5, {'name': 'mc_FM_50_2b', 'dataset_ratio': 0.045, 'single_batch': False, 'soft_sampling': 4})]

#rifare single batch
mnist_consume_2b_p500 = [(CURIOSITY_CONSUME_MODE, 0.5, {'name': 'mc_CONS_50_2b_p500', 'pool_size': 500}),
                    (CURIOSITY_CONSUME_MODE, 0.75, {'name': 'mc_CONS_75_2b_p500', 'pool_size': 500}),
                    (CURIOSITY_CONSUME_MODE, 0.99, {'name': 'mc_CONS_99_2b_p500', 'pool_size': 500})]

full_comp_sb = [(BASELINE, 0.50, {'name': 'full_comp_sb_BL', 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.01, {'name': 'full_comp_sb_1', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.05, {'name': 'full_comp_sb_5', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.10, {'name': 'full_comp_sb_10', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.20, {'name': 'full_comp_sb_20', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.30, {'name': 'full_comp_sb_30', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.40, {'name': 'full_comp_sb_40', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.50, {'name': 'full_comp_sb_50', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.60, {'name': 'full_comp_sb_60', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.70, {'name': 'full_comp_sb_70', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.80, {'name': 'full_comp_sb_80', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.90, {'name': 'full_comp_sb_90', 'dataset_ratio': 0.02, 'single_batch': True}),
                    (CURIOSITY_FULL_MODE, 0.99, {'name': 'full_comp_sb_100', 'dataset_ratio': 0.02, 'single_batch': True})]



runs = full_comp_sb

base_batch_size = 125

notes = sys.argv[2]

fprint(f"##### NAME: {name}")

for i in range(100):
    src_bak_name = f"logs/{name}/{os.path.basename(sys.argv[0])}_{i}"
    if not os.path.exists(src_bak_name):
        shutil.copy(sys.argv[0], src_bak_name)
        fprint("Saved backup source file as", src_bak_name)
        break

fprint(f"NOTES: {notes}")
fprint(f"PARAMS: SEED {SEED}, DEFAULT_SOFT_SAMPLING {DEFAULT_SOFT_SAMPLING}, shuffle {shuffle}")
fprint(f"PARAMS: EARLY_STOP_PATIENCE {EARLY_STOP_PATIENCE}, EARLY_STOP_TOLERANCE {EARLY_STOP_TOLERANCE}")
fprint(f"PARAMS: epochs {epochs}, average_count {average_count}, base_batch_size {base_batch_size}")
fprint("RUNS:", runs)

for i, run in enumerate(runs):

    # all runs use the dataset in the same order (where applicable)
    set_seed(SEED)

    avg_validations = []
    avg_validations_by_iteration = []

    run_name = f"run_{i}_{run[0]}"

    mode, curiosity_ratio, params = run
    if 'name' in params:
        run_name = params['name'] + '_' + run_name

    for ac in range(average_count):

        model = None
        loss_func = None
        K.clear_session()

        #model = create_model()
        model = create_model()

        logdir = f"logs/{name}/test/{name}_{run_name}_avg_{ac}"
        test_writer = tf.summary.FileWriter(logdir)

        logdir = f"logs/{name}/train/{name}_{run_name}_avg_{ac}"
        train_writer = tf.summary.FileWriter(logdir)

        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        #eval_callbacks = [tensorboard_callback]
        eval_callbacks = []

        start = time.time()
        fprint(f"{i} RUN {run} avg_iter: {ac} started.")
        fprint(f"Train params {run[2]}")
        validation, validation_by_iteration, best_accuracies = train(mode, base_batch_size,
                                                                     curiosity_ratio=curiosity_ratio, params=params)
        fprint(f"{i} RUN {run} avg_iter: {ac} done. Elapsed: ", time.time() - start)

        avg_validations.append(validation)
        np.savetxt(f'{root}/data_{name}_{run_name}_{ac}', validation, delimiter=',')

        avg_validations_by_iteration.append(validation_by_iteration)
        np.savetxt(f'{root}/data_{name}_{run_name}_{ac}_by_iteration', validation_by_iteration, delimiter=',')

        test_writer.close()
        train_writer.close()

    min_len = min([len(x) for x in avg_validations])
    cut_avg_validations = [x[:min_len] for x in avg_validations]
    avg_validation = np.mean(cut_avg_validations, axis=0)
    np.savetxt(f'{root}/data_{name}_{run_name}_avg', avg_validation, delimiter=',')

    print("avg len", [len(x) for x in avg_validations_by_iteration])
    min_len = min([len(x) for x in avg_validations_by_iteration])
    cut_avg_validations_by_iteration = [x[:min_len] for x in avg_validations_by_iteration]
    avg_validation_by_iteration = np.mean(cut_avg_validations_by_iteration, axis=0)
    np.savetxt(f'{root}/data_{name}_{run_name}_avg_by_iteration', avg_validation_by_iteration, delimiter=',')

    # add accuracy on tensorboard
    logdir = f"logs/{name}/test/{name}_{run_name}_AVG"
    avg_val_writer = tf.summary.FileWriter(logdir)
    for iter, loss, acc in avg_validation_by_iteration:
        summary = tf.Summary(value=[tf.Summary.Value(tag="average test accuracy",
                                                     simple_value=acc)])
        avg_val_writer.add_summary(summary, iter)
    avg_val_writer.flush()
    avg_val_writer.close()

    print("best_accuracies", best_accuracies)
    best_accuracies = np.asarray(best_accuracies).reshape(-1, 3)
    avg_acc = np.mean(best_accuracies[:, 0])
    avg_epoch = np.mean(best_accuracies[:, 1])
    avg_elapsed = np.mean(best_accuracies[:, 2])
    fprint(f"### Best average accuracy for run {run_name}: {avg_acc} at average epoch {avg_epoch} (avg_elapsed {avg_elapsed})")


