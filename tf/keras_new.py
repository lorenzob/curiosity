from __future__ import print_function

import random
import sys
import time

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, np
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf

batch_size = 100
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
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model


model = create_model()

data_gen = data_generator_mnist(x_train, y_train, batch_size)

def train(marc_mode=False, fix=1):

    count = 0
    validation = []
    for e in range(epochs):

        print("##Epoch", e)
        epoch_count = 0
        for i in range(int(x_train.shape[0] / batch_size)):

            batch_data = next(data_gen)
            images = batch_data[0]
            labels = batch_data[1]

            if marc_mode:
                logits = model.predict(images)

                # print("logits", logits)

                losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                                    logits=logits)

                #print(losses)
                #print("losses", np.mean(tf.keras.backend.eval(losses)))
                losses = tf.keras.backend.eval(losses)
                #print(losses)

                # scaled
                scaled = (losses - min(losses)) / (max(losses) - min(losses))
                lweights = np.square(scaled)

                # norm
                lweights /= np.sum(lweights)

                # try to make it comparable to the other weights set
                lweights *= batch_size

                sample_weights = lweights

                sample_weights = np.ones(shape=(batch_size)) * fix
            else:
                sample_weights = np.ones(shape=(batch_size))
                #print(sample_weights)

            model.fit(images, labels,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=1,
                      sample_weight=sample_weights,
                      shuffle=False,
                      validation_data=(x_test, y_test))

            epoch_count += batch_size
            print("Processed samples", epoch_count)

            if epoch_count % record_steps == 0:
                print("Testing model...")
                loss_acc = model.evaluate(x_test, y_test)
                print("loss_acc", loss_acc)
                validation.append(loss_acc[1])

        count += epoch_count
        print("Total processed samples", count)
        return validation

import matplotlib.pyplot as plt

color = ['b--', 'g', 'r', 'c', 'm', 'y', 'k']

plot_steps = (len(y_train)*epochs) / record_steps

#print(validation)
#fig, ax = plt.subplots()
#ax.grid()
t = np.arange(0, plot_steps)
fig, ax = plt.subplots()
ax.grid()
ax.set_ylim(0.6, 1)

average_count = 1
runs = [(False, None), (True, 0.001), (True, 0.1), (True, 10), (True, 1000)]
#runs = [(False, None)]
name = sys.argv[1]
for i, run in enumerate(runs):

    avg_validations = []
    for ac in range(average_count):

        model = create_model()
        data_gen = data_generator_mnist(x_train, y_train, batch_size)
        validation = train(marc_mode=run[0], fix=run[1])
        avg_validations.append(validation)
        np.savetxt(f'data/data_{name}_{i}_{ac}', validation, delimiter=',')

        ax.plot(t, validation, color[i], alpha=0.2)
        fig.savefig(f"preview_{name}.png", dpi=200)

    avg_validation = np.mean(avg_validations, axis=0)
    np.savetxt(f'data/data_{name}_{i}_avg', avg_validation, delimiter=',')
    ax.plot(t, avg_validation, color[i])

# fig.savefig("all.png", dpi=200)
fig.savefig(f"all_{name}.png", dpi=200)
#np.savetxt(f'data/data_{name}_{i}', avg_validation, delimiter=',')


#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])