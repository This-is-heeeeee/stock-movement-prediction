import os
import math
import json
import sys
import argparse
import numpy as np
import time
from datetime import timedelta
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, \
    Activation, add
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Layer, InputSpec
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.optimizers import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import dataset as dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def build_dataset(data_dir, img_width):
    X, y, tags = dataset.dataset(data_dir)
    nb_classes = len(tags)

    sample_count = len(y)
    train_size = sample_count
    print("train size : {}".format(train_size))
    feature = X
    label = utils.to_categorical(y, nb_classes)
    return feature, label, nb_classes


def build_model(SHAPE, seed=None):
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    # Pooling
    x = Conv2D(32, (3, 3), kernel_initializer='glorot_uniform',
               padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(48, (3, 3), kernel_initializer='glorot_uniform',
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), kernel_initializer='glorot_uniform',
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(96, (3, 3), kernel_initializer='glorot_uniform',
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Flattening
    x = Flatten()(x)

    # Full connectino
    x = Dense(units=256, activation='relu')(x)

    # Dropout
    x = Dropout(0.5)(x)

    x = Dense(units=2, activation='softmax')(x)

    model = Model(input_layer, x)
    print("input_layer : {}".format(input_layer))
    print("x : {}".format(x))

    return model


def main():
    start_time = time.monotonic()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input',
                        help='an input directory of dataset', required=True)
    parser.add_argument('-d', '--dimension',
                        help='an image dimension', type=int, default=48)
    parser.add_argument('-c', '--channel',
                        help='an image channel', type=int, default=3)
    parser.add_argument('-e', '--epochs',
                        help='num of epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size',
                        help='num of batch_size', type=int, default=64)
    parser.add_argument('-o', '--output',
                        help='a result file', type=str, default="result.txt")

    args = parser.parse_args()

    img_width, img_height = args.dimension, args.dimension
    channel = args.channel
    epochs = args.epochs
    batch_size = args.batch_size
    SHAPE = (img_width, img_height, channel)
    bn_axis = 3 if K.image_data_format() == 'tf' else 1

    data_dir = args.input

    print("loading dataset")
    X_train, Y_train, nb_classes = build_dataset(
        "{}/train".format(data_dir), args.dimension)
    X_test, Y_test, nb_classes = build_dataset(
        "{}/test".format(data_dir), args.dimension)
    print("num of classes : {}".format(nb_classes))

    model = build_model(SHAPE)

    model.compile(optimizer=Adam(lr=1.0e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    #   print("X_train:{}\nY_train:{}\n".format(len(X_train), len(Y_train)))

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

    model.save('{}epochs_{}batch_cnn_model_{}.h5'.format(
        epochs, batch_size, data_dir.replace("/", "_")), overwrite=True)

    predicted = model.predict(X_test)
    y_pred = np.argmax(predicted, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_test, y_pred)
    report = classification_report(Y_test, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    if tp == 0: tp = 1
    if tn == 0: tn = 1
    if fp == 0: fp = 1
    if fn == 0: fn = 1

    TPR = float(tp) / (float(tp) + float(fn))  # tp/positive = sensitivity
    FPR = float(fp) / (float(fp) + float(tn))  # fp/negative = 1 - specificity
    accuracy = round((float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn)), 3)  # true/all
    specificity = round(float(tn) / (float(tn) + float(fp)), 3)  # tn/negative
    sensitivity = round(float(tp) / (float(tp) + float(fn)), 3)  # tp/positive
    mcc = round((float(tp) * float(tn) - float(fp) * float(fn)) / math.sqrt((float(tp) + float(fp))
                                                                            * (float(tp) + float(fn))
                                                                            * (float(tn) + float(fp))
                                                                            * (float(tn) + float(fn))), 3)

    f_output = open(args.output, 'a')
    f_output.write('==========\n')
    f_output.write('{}epochs_{}batch_cnn\n'.format(epochs, batch_size))
    f_output.write('TN : {}\n'.format(tn))
    f_output.write('FN : {}\n'.format(fn))
    f_output.write('TP : {}\n'.format(tp))
    f_output.write('FP : {}\n'.format(fp))
    f_output.write('TPR : {}\n'.format(TPR))
    f_output.write('FPR : {}\n'.format(FPR))
    f_output.write('Accuracy : {}\n'.format(accuracy))
    f_output.write('Specificity : {}\n'.format(specificity))
    f_output.write('Sensitivity : {}\n'.format(sensitivity))
    f_output.write('mcc : {}\n'.format(mcc))
    f_output.write('{}'.format(report))
    f_output.write('==========\n')
    f_output.close()
    end_time = time.monotonic()
    print("Duration : {}".format(timedelta(seconds=end_time - start_time)))


if __name__ == "__main__":
    main()