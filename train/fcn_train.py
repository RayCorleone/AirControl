# -*- coding: utf-8 -*-
"""
    @author:    Ray
    @usage:     Train & Save a Model
    @version:   0.1
"""

from __future__ import print_function

import keras
import tensorflow as tf
import numpy as np

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

nb_epochs = 2000

flist  = ['data']
for each in flist:
    fname = each
    x_train, y_train = readucr(fname+'/'+'RIP'+'_TRAIN')
    x_test, y_test = readucr(fname+'/'+'RIP'+'_TEST')

    nb_classes = len(np.unique(y_train))
    batch_size = min(x_train.shape[0]/10, 16)

    print(type(x_train))
    print('nb_classes:',nb_classes)
    print('y_train_ori:',y_train)
    print('y_test_ori:',y_test)

    y_test = (y_test - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
    print('y_train:',y_train)
    print('y_test:',y_test)
    
    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.np_utils.to_categorical(y_test, nb_classes)
    
    print('Y_T:',Y_train)
    print('Y_P:',Y_test)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    # print('x_train_mean:',x_train_mean)
    # print('x_train_std:',x_train_std)

    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_train_mean)/(x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,1,))
    x_test = x_test.reshape(x_test.shape + (1,1,))


    x = keras.layers.Input(x_train.shape[1:])
    # drop_out = Dropout(0.2)(x)
    conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    
    # drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    
    # drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    
    full = keras.layers.GlobalAveragePooling2D()(conv3)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
        
    model = keras.models.Model(inputs=x, outputs=out)
    
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
                        patience=50, min_lr=0.0001)
    
    print("I'm here!!!!")
    hist = model.fit(x_train, Y_train, 
                    batch_size=batch_size, epochs=nb_epochs,
                    verbose=1, validation_data=(x_test,Y_test),
                    callbacks = [reduce_lr])
    
    # Test Predict
    classes = model.predict(x_test, batch_size=20)
    print("Notice: Predict Result")
    print(classes.shape)

    # Save Model
    model_save_path = "./model/keras_2000.h5"
    model.save(model_save_path)

    # Test Load Model
    print("Notice: Reload Predict Result")
    del model
    from keras.models import load_model
    model = load_model(model_save_path)
    classes = model.predict(x_test, batch_size=20)
    print(classes.shape)
    print(classes)