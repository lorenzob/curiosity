from __future__ import print_function

import random
import sys

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, np, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K, Input

import tensorflow as tf


num_classes = 10
epochs = 1
SEED=123

record_steps=1000

def set_seed(seed):
    print("SEED", seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_random_seed(seed)

set_seed(SEED)

def data_generator_mnist(X_test, y_test, batchSize):
    dataset = (X_test, y_test)
    dataset_size = len(X_test)

    i = 0
    while (True):
        if (i + batchSize > dataset_size):

            # i = 0  # simplify?

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
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    return model


#model = create_model()

#data_gen = data_generator_mnist(x_train, y_train, batch_size)

def translate(value, fromMin, fromMax, toMin, toMax):
    # Figure out how 'wide' each range is
    fromSpan = fromMax - fromMin
    toSpan = toMax - toMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - fromMin) / fromSpan

    # Convert the 0-1 range into a value in the right range.
    return toMin + (valueScaled * toSpan)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

BASELINE=0
MARC_MODE=1
CURIOSITY_MODE=2
CURIOSITY_BASELINE=3
CURIOSITY_BASELINE_FULL_SAMPLE=4
CURIOSITY_MODE_SINGLE_BATCH=5

def train(mode, curiosity_ratio=1):

    base_batch_size = 100
    k = int(base_batch_size * curiosity_ratio)

    if mode in [BASELINE, MARC_MODE]:
        batch_size = base_batch_size + k
    else:
        batch_size = base_batch_size

    data_gen = data_generator_mnist(x_train, y_train, batch_size)

    count = 0
    validation = []
    validation_by_sample_count = []
    sample_count = 0
    for e in range(epochs):

        print("##Epoch", e, batch_size)
        epoch_count = 0
        for i in range(int(x_train.shape[0] / batch_size)):

            batch_data = next(data_gen)
            images = batch_data[0]
            labels = batch_data[1]

            if mode == BASELINE:

                sample_weights = np.ones(shape=batch_size)

                assert len(labels) == batch_size

                model.fit(images, labels,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          sample_weight=sample_weights,
                          shuffle=False)
                sample_count += len(labels)

            elif mode == MARC_MODE:

                y_true = Input(shape=(10,))
                ce = K.categorical_crossentropy(y_true, model.output)
                func = K.function(model.inputs + [y_true], [ce])
                losses = func([images, labels])[0]

                # min/max 1 to 10
                proc_losses = translate(losses, min(losses), max(losses), 1, 50)

                #proc_losses = np.square(proc_losses)

                #proc_losses = softmax(losses)
                #proc_losses = translate(proc_losses, min(proc_losses), max(proc_losses), 1, 100)

                # norm
                #lweights /= np.sum(lweights)

                # try to make it comparable to the other weights set
                #lweights *= batch_size

                #sample_weights = lweights

                # temp
                #sample_weights = np.ones(shape=(batch_size)) * fix
                sample_weights = proc_losses

                split_batches = True
                if split_batches:

                    first_images = images[:base_batch_size]
                    first_labels = labels[:base_batch_size]
                    first_weights = sample_weights[:base_batch_size]

                    assert len(first_labels) == base_batch_size
                    model.fit(first_images, first_labels,
                              batch_size=batch_size,
                              epochs=1,
                              verbose=1,
                              sample_weight=first_weights,
                              shuffle=False)
                    sample_count += len(first_labels)

                    second_images = images[base_batch_size:]
                    second_labels = labels[base_batch_size:]
                    second_weights = sample_weights[base_batch_size:]

                    assert len(second_labels) == k
                    model.fit(second_images, second_labels,
                              batch_size=k,
                              epochs=1,
                              verbose=1,
                              sample_weight=second_weights,
                              shuffle=False)
                    sample_count += len(second_labels)

                else:
                    assert len(labels) == batch_size

                    model.fit(images, labels,
                              batch_size=batch_size,
                              epochs=1,
                              verbose=1,
                              sample_weight=sample_weights,
                              shuffle=False)
                    sample_count += len(labels)

            elif mode == CURIOSITY_MODE_SINGLE_BATCH:

                # compute losses
                y_true = Input(shape=(10,))
                ce = K.categorical_crossentropy(y_true, model.output)
                func = K.function(model.inputs + [y_true], [ce])
                losses = func([images, labels])[0]

                worst = np.argpartition(losses, -k)
                retry_idx = worst[-k:]

                retry_images = batch_data[0][retry_idx]
                retry_labels = batch_data[1][retry_idx]

                joined_images = np.append(images, retry_images, axis=0)
                joined_labels = np.append(labels, retry_labels, axis=0)

                assert len(joined_labels) == batch_size+k

                model.fit(joined_images, joined_labels,
                          batch_size=batch_size+k,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(joined_labels)

            elif mode == CURIOSITY_MODE:

                assert len(labels) == batch_size

                model.fit(images, labels,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(labels)

                # compute losses before or after fit?

                # compute losses
                y_true = Input(shape=(10,))
                ce = K.categorical_crossentropy(y_true, model.output)
                func = K.function(model.inputs + [y_true], [ce])
                losses = func([images, labels])[0]

                worst = np.argpartition(losses, -k)
                retry_idx = worst[-k:]

                retry_images = batch_data[0][retry_idx]
                retry_labels = batch_data[1][retry_idx]

                assert len(retry_labels) == k

                model.fit(retry_images, retry_labels,
                          batch_size=k,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(retry_labels)

            elif mode == CURIOSITY_BASELINE:

                assert len(labels) == batch_size

                model.fit(images, labels,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(labels)

                # sample over the same batch
                indexes = np.arange(batch_size)
                retry_idx = np.random.choice(indexes, size=k, replace=False)
                retry_images = batch_data[0][retry_idx]
                retry_labels = batch_data[1][retry_idx]

                assert len(retry_labels) == k

                model.fit(retry_images, retry_labels,
                          batch_size=k,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(retry_labels)

            elif mode == CURIOSITY_BASELINE_FULL_SAMPLE:

                assert len(labels) == batch_size

                model.fit(images, labels,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(labels)

                # sample over the whole training set
                indexes = np.arange(x_train.shape[0])
                retry_idx = np.random.choice(indexes, size=k, replace=False)
                retry_images = x_train[retry_idx]
                retry_labels = y_train[retry_idx]

                assert len(retry_labels) == k

                model.fit(retry_images, retry_labels,
                          batch_size=k,
                          epochs=1,
                          verbose=1,
                          shuffle=False)
                sample_count += len(retry_labels)

            else:
                raise Exception("Unsupported mode " + str(mode))
                #print(sample_weights)

            epoch_count += batch_size
            print("Processed samples", epoch_count)

            if epoch_count % record_steps == 0:
                print("Testing model...")
                loss_acc = model.evaluate(x_test, y_test)
                print("loss_acc", loss_acc)

                validation.append(loss_acc[1])
                validation_by_sample_count.append((sample_count, loss_acc[0], loss_acc[1]))

        count += epoch_count
        print("Total processed samples", count)
        return validation, validation_by_sample_count

import matplotlib.pyplot as plt

color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b--', 'g--', 'r--', 'c--', 'm--', 'y--', 'k--']

plot_steps = (len(y_train)*epochs) / record_steps

#print(validation)
#fig, ax = plt.subplots()
#ax.grid()
t = np.arange(0, plot_steps)
fig, ax = plt.subplots()
ax.grid()
ax.set_ylim(0.6, 1)

average_count = 2
#runs = [(False, None), (True, 0.001), (True, 0.1), (True, 10), (True, 1000)]
#runs = [(MARC_MODE, 0.25)]

runs = [(CURIOSITY_MODE, 0.25)]

#runs = [(BASELINE, ), (CURIOSITY_BASELINE, ), (CURIOSITY_BASELINE_FULL_SAMPLE, ),
#        (CURIOSITY_MODE, ), (CURIOSITY_MODE_SINGLE_BATCH, )]

#runs = [(CURIOSITY_MODE, 0.1), (CURIOSITY_MODE, 0.2), (CURIOSITY_MODE, 0.3), (CURIOSITY_MODE, 0.4),
#        (CURIOSITY_MODE, 0.5), (CURIOSITY_MODE, 0.6), (CURIOSITY_MODE, 0.7), (CURIOSITY_MODE, 0.8),
#        (CURIOSITY_MODE, 0.9), (CURIOSITY_MODE, 1),]


# runs = [(True, 1)]
name = sys.argv[1]
for i, run in enumerate(runs):

    avg_validations = []
    avg_validations_by_sample_count = []
    for ac in range(average_count):

        model = create_model()
        validation, validation_by_sample_count = train(mode=run[0], curiosity_ratio=run[1])
        #print(validation)
        #print(validation_by_sample_count)
        #print("aaa")
        avg_validations.append(validation)
        np.savetxt(f'data/data_{name}_{i}_{ac}', validation, delimiter=',')

        avg_validations_by_sample_count.append(validation_by_sample_count)

        ax.plot(t, validation, color[i], alpha=0.2)
        fig.savefig(f"preview_{name}.png", dpi=200)

    avg_validation = np.mean(avg_validations, axis=0)
    ax.plot(t, avg_validation, color[i])
    np.savetxt(f'data/data_{name}_{i}_avg', avg_validation, delimiter=',')

    avg_validation_by_sample_count = np.mean(avg_validations_by_sample_count, axis=0)
    np.savetxt(f'data/data_{name}_{i}_avg_by_sample_count', avg_validation_by_sample_count, delimiter=',')

# fig.savefig("all.png", dpi=200)
fig.savefig(f"all_{name}.png", dpi=200)
#np.savetxt(f'data/data_{name}_{i}', avg_validation, delimiter=',')

t = avg_validation_by_sample_count.T[0]
fig2, ax2 = plt.subplots()
ax2.grid()
ax2.set_ylim(0.6, 1)
ax2.set_xlim(0, 60000)
ax2.plot(t, avg_validation_by_sample_count.T[2], color[i])
#ax2.plot(t, avg_validation_by_sample_count.T[1], color[i])
fig2.savefig(f"all_{name}_by_samples.png", dpi=200)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])