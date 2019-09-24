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

	def mcc_metric(y_true, y_pred):
		predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
		true_pos = tf.math.count_nonzero(predicted * y_true)
		true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
		false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
		false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
		x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
		return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)

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

			model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(train_x.shape[1],)))

			for i in range(0, best_layer):
				model.add(tf.keras.layers.Dense(best_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
				model.add(tf.keras.layers.Dropout(d))

			model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

			patience = int(config.get('MLPNN', 'Patience'))
			batch_size = int(config.get('MLPNN', 'BatchSize'))	
			learning_rate = float(config.get('MLPNN', 'LearningRate'))

			if train_y.shape[1] == 2:
				es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
				model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
					metrics=[auc_metric, mcc_metric])
			else:
				es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
				model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
					metrics=[mcc_metric])

			if len(train_weights) == 0:
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)
			else:
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1, 
					sample_weight = train_weights)
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10, sample_weight = train_weights)
			
			if train_y.shape[1] == 2:
				stat = model.evaluate(test_x, test_y)[1]
			else:
				stat = model.evaluate(test_x, test_y)[1]
			if stat > best_stat:
				best_stat = stat
				best_drop = d
				best_l2 = l
			tf.reset_default_graph()

	for l in num_layer:
		for n in num_nodes:

			reg = tf.keras.regularizers.l2(best_l2)
			model = tf.keras.Sequential()

			model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(input_len,)))

			for i in range(0, l):
				model.add(tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
				model.add(tf.keras.layers.Dropout(best_drop))

			model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

			patience = int(config.get('MLPNN', 'Patience'))
			batch_size = int(config.get('MLPNN', 'BatchSize'))


			if train_y.shape[1] == 2:
				es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
				model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
					metrics=[auc_metric, mcc_metric])
			else:
				es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
				model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
					metrics=[mcc_metric])

			if len(train_weights) == 0:
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)
			else:
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1, 
					sample_weight = train_weights)
				model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10, sample_weight = train_weights)

			if train_y.shape[1] == 2:
				stat = model.evaluate(test_x, test_y)[1]
			else:
				stat = model.evaluate(test_x, test_y)[1]
			if stat > best_stat:
				best_stat = stat
				best_layer = l
				best_nodes = n
			tf.reset_default_graph()
	return best_layer, best_nodes, best_l2, best_drop

def tune_PopPhy(train, test, config, train_weights=[]):
		
		def mcc_metric(y_true, y_pred):
			predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
			true_pos = tf.math.count_nonzero(predicted * y_true)
			true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
			false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
			false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
			x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
			return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)

		def auc_metric(y_true, y_pred):
			return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)
		
		num_row, num_col = (train[0].shape[1], train[0].shape[2])
		train_x, train_y = train
		test_x, test_y = test
		train_x = np.expand_dims(train_x, -1)
		test_x = np.expand_dims(test_x, -1)

		num_class = train_y.shape[1]
		num_row, num_col = train_x.shape[1], train_x.shape[2]
		best_num_kernel = 64
		best_kernel_height = 3
		best_kernel_width = 7
		best_num_fc_nodes = 64
		best_num_cnn_layers = 2
		best_num_fc_layers = 1
		best_l2 = 0.001
		best_drop = 0.3

		l2_list = [0.001, 0.0001]
		drop_list = [0.3, 0.5]
		
		kernel_list = [32,64,128]
		cnn_layer_list = [1,2,3]
		
		kernel_w_list = [5,7,9]
		kernel_h_list = [2,3,4]
		
		fc_layer_list = [1,2]
		fc_node_list = [32,64,128]

		patience = int(config.get('PopPhy', 'Patience'))
		learning_rate = float(config.get('PopPhy', 'LearningRate'))
		batch_size = int(config.get('PopPhy', 'BatchSize'))
		best_stat = 0

		for l in l2_list:
			for d in drop_list:
				reg = tf.keras.regularizers.l2(l)
				model = tf.keras.Sequential()
	
				model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(train_x.shape[1], train_x.shape[2], 1)))
	
				for i in range(0, best_num_cnn_layers):
					model.add(tf.keras.layers.Conv2D(filters=best_num_kernel, kernel_size=(best_kernel_height, best_kernel_width),
						activation='relu', bias_regularizer=reg, kernel_regularizer=reg, name="conv_"+str(i)))
					model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,2)))


				model.add(tf.keras.layers.Flatten())
				model.add(tf.keras.layers.Dropout(d))

				for i in range(0, best_num_fc_layers):
					model.add(tf.keras.layers.Dense(best_num_fc_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
					model.add(tf.keras.layers.Dropout(d))

					model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))
			
				if train_y.shape[1] == 2:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[auc_metric, mcc_metric])
				else:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[mcc_metric])

				if len(train_weights) == 0:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)
				else:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1,
						sample_weight = train_weights)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10, sample_weight = train_weights)
	
				if train_y.shape[1] == 2:
					stat = model.evaluate(test_x, test_y)[1]
				else:
					stat = model.evaluate(test_x, test_y)[1]
				if stat > best_stat:
					best_stat = stat
					best_drop = d
					best_l2 = l	
				tf.reset_default_graph()

		for k in kernel_list:
			for l in cnn_layer_list:
				reg = tf.keras.regularizers.l2(best_l2)
				model = tf.keras.Sequential()

				model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(num_row, num_col, 1)))

				for i in range(0, l):
					model.add(tf.keras.layers.Conv2D(filters=k, kernel_size=(best_kernel_height, best_kernel_width),
						activation='relu', bias_regularizer=reg, kernel_regularizer=reg))
					model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,2)))


				model.add(tf.keras.layers.Flatten())
				model.add(tf.keras.layers.Dropout(best_drop))

				for i in range(0, best_num_fc_layers):
					model.add(tf.keras.layers.Dense(best_num_fc_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
					model.add(tf.keras.layers.Dropout(best_drop))

					model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg))
			
				if train_y.shape[1] == 2:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[auc_metric, mcc_metric])
				else:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[mcc_metric])
	
				if len(train_weights) == 0:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)
				else:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1,
						sample_weight = train_weights)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10, sample_weight = train_weights)
	
				if train_y.shape[1] == 2:
					stat = model.evaluate(test_x, test_y)[1]
				else:
					stat = model.evaluate(test_x, test_y)[1]
				if stat > best_stat:
					best_stat = stat
					best_num_kernel = k
					best_num_cnn_layers = l
				tf.reset_default_graph()

		for w in kernel_w_list:
			for h in kernel_h_list:

				reg = tf.keras.regularizers.l2(best_l2)
				model = tf.keras.Sequential()

				model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(num_row, num_col, 1)))

				for i in range(0, best_num_cnn_layers):
					model.add(tf.keras.layers.Conv2D(filters=best_num_kernel, kernel_size=(h, w),
						activation='relu', bias_regularizer=reg, kernel_regularizer=reg, name="conv_"+str(i)))
					model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,2)))


				model.add(tf.keras.layers.Flatten())
				model.add(tf.keras.layers.Dropout(best_drop))

				for i in range(0, best_num_fc_layers):
					model.add(tf.keras.layers.Dense(best_num_fc_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
					model.add(tf.keras.layers.Dropout(d))

					model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg))
			
				if train_y.shape[1] == 2:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[auc_metric, mcc_metric])
				else:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[mcc_metric])

				if len(train_weights) == 0:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)
				else:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1,
						sample_weight = train_weights)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10, sample_weight = train_weights)

				if train_y.shape[1] == 2:
					stat = model.evaluate(test_x, test_y)[1]
				else:
					stat = model.evaluate(test_x, test_y)[1]
				if stat > best_stat:
					best_stat = stat
					best_kernel_height = h
					best_kernel_weight = w
				tf.reset_default_graph()

		for l in fc_layer_list:
			for n in fc_node_list:

				reg = tf.keras.regularizers.l2(best_l2)
				model = tf.keras.Sequential()

				model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(num_row, num_col, 1)))

				for i in range(0, best_num_cnn_layers):
					model.add(tf.keras.layers.Conv2D(filters=best_num_kernel, kernel_size=(best_kernel_height, best_kernel_width),
						activation='relu', bias_regularizer=reg, kernel_regularizer=reg, name="conv_"+str(i)))
					model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,2)))


				model.add(tf.keras.layers.Flatten())
				model.add(tf.keras.layers.Dropout(best_drop))

				for i in range(0, l):
					model.add(tf.keras.layers.Dense(n, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
					model.add(tf.keras.layers.Dropout(best_drop))

					model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg))
			
				if train_y.shape[1] == 2:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[auc_metric, mcc_metric])
				else:
					es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True)
					model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy',
						metrics=[mcc_metric])
	
				if len(train_weights) == 0:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10)
				else:
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1,
						sample_weight = train_weights)
					model.fit(train_x, train_y, batch_size=batch_size, verbose=0, epochs=10, sample_weight = train_weights)
	
				if train_y.shape[1] == 2:
					stat = model.evaluate(test_x, test_y)[1]
				else:
					stat = model.evaluate(test_x, test_y)[1]
				if stat > best_stat:
					best_stat = stat
					best_num_fc_layers = l
					best_num_fc_nodes = n
				tf.reset_default_graph()

		return best_num_kernel, best_kernel_height, best_kernel_width, best_num_cnn_layers, best_num_fc_layers, best_num_fc_nodes, best_l2, best_drop 
