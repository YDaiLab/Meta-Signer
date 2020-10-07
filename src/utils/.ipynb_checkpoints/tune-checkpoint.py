import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from utils.popphy_io import get_stat, get_stat_dict

def tune_mlpnn(train, test, config, train_weights=[]):

    train_x, train_y = train
    test_x, test_y = test
    num_class = train_y.shape[1]
    input_len = train_x.shape[1]

    def auc_metric(y_true, y_pred):
        return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)

    dropout = [0.1, 0.3, 0.5]
    l2_grid = [0.01, 0.001, 0.0001]
    num_layer = [1,2]
    num_nodes = [32,64,128]
    
    best_l2 = 0.0001
    best_drop = 0.5
    best_layer = 2
    best_nodes = 128

    best_stat = 0

    for d in dropout:
        for l in l2_grid:
            reg = tf.keras.regularizers.l2(l)
            model = tf.keras.Sequential()
            
            for i in range(0, best_layer):
                model.add(tf.keras.layers.Dense(best_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
                model.add(tf.keras.layers.Dropout(d))

            model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

            patience = int(config.get('MLPNN', 'Patience'))
            batch_size = int(config.get('MLPNN', 'BatchSize'))
            learning_rate = float(config.get('MLPNN', 'LearningRate'))

            es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')
            print(train_x)
            print(train_y)
            model.fit(train_x, train_y, batch_size=batch_size, verbose=1, epochs=1000, callbacks=[es_cb], validation_split=0.1)
            model.fit(train_x, train_y, batch_size=batch_size, verbose=1, epochs=10)

            probs = model.predict(test_x)
            preds = np.argmax(probs, axis=1)
            stat = get_stat_dict(np.argmax(test_y, axis=1), probs, preds)            

            if stat["AUC"] > best_stat:
                best_stat = stat["AUC"]
                best_drop = d
                best_l2 = l
            tf.reset_default_graph()
            tf.keras.backend.clear_session()
            
    for l in num_layer:
        for n in num_nodes:

            reg = tf.keras.regularizers.l2(best_l2)
            model = tf.keras.Sequential()


            for i in range(0, l):
                model.add(tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
                model.add(tf.keras.layers.Dropout(best_drop))

            model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

            patience = int(config.get('MLPNN', 'Patience'))
            batch_size = int(config.get('MLPNN', 'BatchSize'))


            es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

            model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
            model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)


            probs = model.predict(test_x)
            preds = np.argmax(probs, axis=1)
            stat = get_stat_dict(np.argmax(test_y, axis=1), probs, preds)            

            if stat["AUC"] > best_stat:
                best_stat = stat["AUC"]
                best_layer = l
                best_nodes = n
            tf.reset_default_graph()
            tf.keras.backend.clear_session()
            
    return best_layer, best_nodes, best_l2, best_drop
