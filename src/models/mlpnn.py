import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.popphy_io import get_stat, get_stat_dict
from utils.feature_map_analysis import get_feature_rankings_mlpnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def PopPhyMLPNN(num_class, num_features, num_layers, num_nodes, lamb, drop=0.0):
	X = tf.placeholder("float", [None, num_features], name="X")
	Y = tf.placeholder("float", [None], name="Y")
	Noise = tf.placeholder("float", [None, num_features], name="Noise")
	CW = tf.placeholder("float", [None], name="CW")
	training = tf.placeholder(tf.bool)
	batch_size = tf.placeholder(tf.int32)
	lab = tf.cast(Y, dtype=tf.int32)
	labels_oh = tf.one_hot(lab, depth=num_class)

	regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)


	with tf.variable_scope("MLPNN", reuse=False):
	
		if num_layers >= 1:
			fc = tf.layers.dense(X, num_nodes, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=False, name="Layer1")
			fc = tf.layers.dropout(fc, rate=drop, training=training)
		
		if num_layers >= 2:
			fc = tf.layers.dense(fc, num_nodes, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer, use_bias=False, name="Layer2")
			fc= tf.layers.dropout(fc, rate=drop, training=training)

		out = tf.layers.dense(fc, num_class, kernel_regularizer=regularizer, bias_regularizer=regularizer, name="Out")

	prob = tf.nn.softmax(out, axis=1)

	with tf.name_scope("Loss"):
		ce_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_oh, logits=out, weights=CW)
		losses = ce_loss + tf.losses.get_regularization_loss()
		cost = tf.reduce_sum(losses)

	return {'ce_cost': ce_loss, 'cost': cost, 'x':X, 'y':Y, 'cw':CW, 'prob':prob, 'pred':prob[:,1], 'training':training, 'noise':Noise, 'batch_size':batch_size}


def train(train, test, config, params, metric, label_set, seed=42):

	learning_rate = float(config.get('MLPNN', 'LearningRate'))
	batch_size = int(config.get('MLPNN', 'BatchSize'))
	
	num_layers = params["Num_Layers"]
	num_nodes = params["Num_Nodes"]
	lamb = params["L2_Lambda"]
	drop = params["Dropout_Rate"]
	max_epoch = params["Max_Epoch"]

	model_num = 1
	
	train_x, train_y = train
	test_x, test_y = test
	
	num_class = len(label_set)
	
	train_x = np.array(train_x)
	test_x = np.array(test_x)
	
	num_features = train_x.shape[1]
	
	test_preds = np.array([])

	c_prob = [1] * num_class
	for l in np.unique(train_y):
		c_prob[int(l)] = float( float(len(train_y))/ (2.0 * float((np.sum(train_y == l)))))

	train_x = np.array(train_x)
	train_y = np.array(train_y)
	train_weights = []
	test_weights = []
			
	for i in train_y:
		train_weights.append(c_prob[int(i)])

	for i in test_y:
		test_weights.append(c_prob[int(i)])
	
	train_var = np.var(train_x,  axis=0)
			
	train_weights = np.array(train_weights)
	test_weights = np.array(test_weights)
	
	final_test_probs = []
	scores = []
	train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
	train_dataset = train_dataset.shuffle(1000000).batch(batch_size)

	train_iterator = train_dataset.make_initializable_iterator()
	next_train_element = train_iterator.get_next()

	test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_weights))
	test_dataset = test_dataset.batch(1024)

	test_iterator = test_dataset.make_initializable_iterator()
	next_test_element = test_iterator.get_next()

	model = PopPhyMLPNN(num_class, num_features, num_layers, num_nodes, lamb, drop=drop)

	with tf.name_scope("Train"):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	feature_scores = {}
	feature_rankings = {}
	with tf.Session() as sess:
		sess.run(init)
		i=0

		num_training_samples = train_x.shape[0]
				
		while True:
			i += 1
			training_loss = 0
			num_training_batches = 0
			num_test_batches = 0
						
			sess.run(train_iterator.initializer)
						
			training_y_list = []
			training_prob_list = []
			while True:
				try:
					batch_x, batch_y, batch_cw = sess.run(next_train_element)
					size = batch_x.shape[0]
					noise = np.random.normal(0, 0.1, list(batch_x.shape))
					_, l, prob, pred = sess.run([optimizer, model['ce_cost'], model['prob'], model['pred']], 
						feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1),
						model['cw']:batch_cw, model['training']:True, model['noise']:noise, model['batch_size']:size})
					training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
					if len(training_prob_list) == 0:
						training_prob_list = prob
					else:
						training_prob_list = np.concatenate((training_prob_list, prob))

					training_loss += l
					num_training_batches += 1

				except tf.errors.OutOfRangeError:
					training_stat = get_stat(training_y_list, training_prob_list, metric)
					training_loss = training_loss/num_training_batches
					break

			sess.run(test_iterator.initializer)

			test_y_list = []
			test_prob_list = []
				
			while True:
				try:
					batch_x, batch_y, batch_cw = sess.run(next_test_element)
					noise = np.zeros(batch_x.shape)
					size = batch_x.shape[0]
					l, prob, pred, y_out = sess.run([model['ce_cost'], model['prob'], model['pred'], model['y']],
						feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw,
						model['training']:False, model['noise']:noise, model['batch_size']:size})
					test_y_list = list(test_y_list) + list(y_out)
					if len(test_prob_list) == 0:
						test_prob_list = prob
					else:
						test_prob_list = np.concatenate((test_prob_list, prob))

				except tf.errors.OutOfRangeError:
					test_stat = get_stat(test_y_list, test_prob_list, metric)
					break

			if i == max_epoch:
				weight_list = []
				with tf.variable_scope('MLPNN', reuse=True):
					for l in range(1, num_layers+1):
						w = tf.get_variable('Layer' + str(l)+"/kernel").eval()
						weight_list.append(w)
					w = tf.get_variable('Out/kernel').eval()
					weight_list.append(w)
					eval_train = train_x
					eval_y = train_y
					eval_l = np.argmax(prob, axis=1)
				scores = get_feature_rankings_mlpnn(weight_list)
				final_test_probs = test_prob_list
				break
	tf.reset_default_graph()	
	test_stat_dict = get_stat_dict(test_y, final_test_probs)
	if num_class == 2:
		fpr, tpr, thresh = roc_curve(test_y, final_test_probs[:,1])
	else:
		fpr, tpr, thresh = 0, 0, 0
	return test_stat_dict, tpr, fpr, thresh, scores, final_test_probs

def tune(train, config, seed=42):

	num_models = int(config.get('MLPNN', 'ValidationModels'))
	learning_rate = float(config.get('MLPNN', 'LearningRate'))
	batch_size = int(config.get('MLPNN', 'BatchSize'))
	max_patience = int(config.get('MLPNN', 'Patience'))	
	
	train_target = 0
	best_model_stat = 0
	mean_cost_target = 0
	best_num_layers = 2
	best_num_nodes = 128
	best_lambda = 0.1
	best_drop = 0.3
	max_epoch = 0
	
	x, y = train
	
	
	model_run = 0
	num_class = len(np.unique(y))

	if num_class == 2:
		metric = "AUC"
	else:
		metric = "MCC"
	
	x = np.array(x)
	num_features = x.shape[1]
	
	skf = StratifiedKFold(n_splits=num_models, random_state=42, shuffle=False)
	
	num_layers_list = [1, 2]
	num_nodes_list = [8, 64]
	lamb_list = [0.01, 0.001]
	drop_list = [0.25, 0.5]
		
	for num_layers in num_layers_list:
		for num_nodes in num_nodes_list:
			
			model_num = 0
			total_training_cost = 0
			total_validation_stat = 0
			total_training_stat = 0
			drop = best_drop
			lamb = best_lambda
			total_avg_epoch = 0
				
			for train_index, validation_index in skf.split(x, y):
				model_num += 1
				patience = max_patience

				train_x, validation_x = x[train_index,:], x[validation_index,:]
				train_y, validation_y = y[train_index], y[validation_index]
				
				scaler = StandardScaler().fit(train_x)
				train_x = scaler.transform(train_x)
				validation_x = scaler.transform(validation_x)
	
				train_x = np.array(train_x, dtype=np.float32)
				validation_x = np.array(validation_x, dtype=np.float32)
				

				c_prob = [1] * num_class
				for l in np.unique(train_y):
					c_prob[int(l)] = float( float(len(train_y))/ (2.0 * float((np.sum(train_y == l)))))
				train_weights = []
				validation_weights = []
				full_weights = []
						
				for i in train_y:
					train_weights.append(c_prob[int(i)])

				for i in validation_y:
					validation_weights.append(c_prob[int(i)])
							
				for i in y:
					full_weights.append(c_prob[int(i)])
							
				train_var = np.var(train_x,  axis=0)

				train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
				train_dataset = train_dataset.shuffle(1000000).batch(batch_size)

				valid_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y, validation_weights))
				valid_dataset = valid_dataset.batch(1024)
		
				train_iterator = train_dataset.make_initializable_iterator()
				next_train_element = train_iterator.get_next()

				valid_iterator = valid_dataset.make_initializable_iterator()
				next_valid_element = valid_iterator.get_next()
		
				model = PopPhyMLPNN(num_class, num_features, num_layers, num_nodes, best_lambda, drop=best_drop)

				with tf.name_scope("Train"):
					optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

				init = tf.global_variables_initializer()
				saver = tf.train.Saver()

				with tf.Session() as sess:
					sess.run(init)
					i=0
					best_validation_loss = 10000
					best_training_loss = 0
					best_validation_stat = 0
					best_training_stat = 0
					avg_epoch = 0
					
					num_training_samples = train_x.shape[0]
					num_validation_samples = validation_x.shape[0]
							
					while True:
						i += 1
						training_loss = 0
						validation_loss = 0
						test_loss = 0
						num_training_batches = 0
						num_validation_batches = 0
						num_test_batches = 0
								
						sess.run(train_iterator.initializer)
								
						training_y_list = []
						training_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_train_element)
								size = batch_x.shape[0]
								noise = np.random.normal(0, 0.1, list(batch_x.shape))
								_, l, pred, prob= sess.run([optimizer, model['ce_cost'], model['pred'], model['prob']], 
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1),model['cw']:batch_cw, 
									model['training']:True, model['noise']:noise, model['batch_size']:size})
								training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
								training_loss += l
								if len(training_prob_list) == 0:
									training_prob_list = prob
								else:
									training_prob_list = np.concatenate((training_prob_list, prob))
								num_training_batches += 1

							except tf.errors.OutOfRangeError:
								training_stat = get_stat_dict(training_y_list, training_prob_list)[metric]
								training_loss = training_loss/num_training_batches
								break
								
						sess.run(valid_iterator.initializer)
								
						validation_y_list = []
						validation_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_valid_element)
								noise = np.zeros(batch_x.shape)
								size = batch_x.shape[0]
								l, pred, prob, y_out = sess.run([model['ce_cost'], model['pred'], model['prob'], model['y']],
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw,
									model['training']:False, model['noise']:noise, model['batch_size']:size})
								validation_y_list = list(validation_y_list) + list(y_out)
								validation_loss += l
								if len(validation_prob_list) == 0:
									validation_prob_list = prob
								else:
									validation_prob_list = np.concatenate((validation_prob_list, prob))
								num_validation_batches += 1

							except tf.errors.OutOfRangeError:
								patience -= 1
								validation_loss = validation_loss/num_validation_batches
								validation_stat = get_stat_dict(validation_y_list, validation_prob_list)[metric]
								if i > 5 and (validation_loss < best_validation_loss or best_validation_stat < validation_stat):
									best_validation_stat = validation_stat
									best_validation_loss = validation_loss
									avg_epoch = i
									patience = max_patience
								break
										
						if patience == 0 or i > 1000:
							total_training_cost += best_training_loss
							total_training_stat += best_training_stat
							total_validation_stat += best_validation_stat
							total_avg_epoch += avg_epoch
							break
				tf.reset_default_graph()
					
			total_validation_stat = total_validation_stat/float(num_models)
			total_training_cost = total_training_cost/float(num_models)
			total_training_stat = total_training_stat/float(num_models)
			total_avg_epoch = total_avg_epoch/float(num_models)
			
			if total_validation_stat > best_model_stat:
				best_model_stat = total_validation_stat
				train_target = total_training_cost
				best_num_layers = num_layers
				best_num_nodes = num_nodes
				max_epoch = np.round(total_avg_epoch)
		
	for lamb in lamb_list:
		for drop in drop_list:
			
			model_num = 0
			total_training_cost = 0
			total_validation_stat = 0
			total_training_stat = 0
			total_avg_epoch = 0

			for train_index, validation_index in skf.split(x, y):
				model_num += 1
				patience = max_patience

				train_x, validation_x = x[train_index,:], x[validation_index,:]
				train_y, validation_y = y[train_index], y[validation_index]
						
				train_x = np.array(train_x, dtype=np.float32)
				validation_x = np.array(validation_x, dtype=np.float32)
						
				c_prob = [1] * num_class
				for l in np.unique(train_y):
					c_prob[int(l)] = float( float(len(train_y))/ (2.0 * float((np.sum(train_y == l)))))
						
				train_weights = []
				validation_weights = []
				full_weights = []
						
				for i in train_y:
					train_weights.append(c_prob[int(i)])

				for i in validation_y:
					validation_weights.append(c_prob[int(i)])
							
				for i in y:
					full_weights.append(c_prob[int(i)])
							
				train_var = np.var(train_x,  axis=0)

				train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
				train_dataset = train_dataset.shuffle(1000000).batch(batch_size)

				valid_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y, validation_weights))
				valid_dataset = valid_dataset.batch(1024)

				train_iterator = train_dataset.make_initializable_iterator()
				next_train_element = train_iterator.get_next()

				valid_iterator = valid_dataset.make_initializable_iterator()
				next_valid_element = valid_iterator.get_next()


				model = PopPhyMLPNN(num_class, num_features, best_num_layers, best_num_nodes, lamb, drop=drop)

				with tf.name_scope("Train"):
					optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

				init = tf.global_variables_initializer()
				saver = tf.train.Saver()

				with tf.Session() as sess:
					sess.run(init)
					i=0
					best_validation_loss = 10000
					best_training_loss = 0
					best_validation_stat = 0
					best_training_stat = 0
					avg_epoch = 0
					
					num_training_samples = train_x.shape[0]
					num_validation_samples = validation_x.shape[0]
							
					while True:
						i += 1
						training_loss = 0
						validation_loss = 0
						test_loss = 0
						num_training_batches = 0
						num_validation_batches = 0
						num_test_batches = 0
						
						sess.run(train_iterator.initializer)
								
						training_y_list = []
						training_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_train_element)
								size = batch_x.shape[0]
								noise = np.random.normal(0, 0.1, list(batch_x.shape))
								_, l, pred, prob = sess.run([optimizer, model['ce_cost'], model['pred'], model['prob']], 
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw, 
									model['training']:True, model['noise']:noise, model['batch_size']:size})
								training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
								if len(training_prob_list) == 0:
									training_prob_list = prob
								else:
									training_prob_list = np.concatenate((training_prob_list, prob))
								training_loss += l
								num_training_batches += 1

							except tf.errors.OutOfRangeError:
								training_stat = get_stat_dict(training_y_list, training_prob_list)[metric]
								training_loss = training_loss/num_training_batches
								break
								
						sess.run(valid_iterator.initializer)
								
						validation_y_list = []
						validation_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_valid_element)
								noise = np.zeros(batch_x.shape)
								size = batch_x.shape[0]
								l, pred, prob, y_out = sess.run([model['ce_cost'], model['pred'], model['prob'], model['y']],
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw,
									model['training']:False, model['noise']:noise, model['batch_size']:size})
								validation_y_list = list(validation_y_list) + list(y_out)
								if len(validation_prob_list) == 0:
									validation_prob_list = prob
								else:
									validation_prob_list = np.concatenate((validation_prob_list, prob))
								num_validation_batches += 1
								validation_loss += l

							except tf.errors.OutOfRangeError:
								patience -= 1
								validation_loss = validation_loss/num_validation_batches
								validation_stat = get_stat_dict(validation_y_list, validation_prob_list)[metric]
								if i > 5 and (validation_loss < best_validation_loss or best_validation_stat < validation_stat):
									best_validation_stat = validation_stat
									best_validation_loss = validation_loss
									avg_epoch = i
									patience = max_patience

								break
										
						if patience == 0 or i > 1000:
							total_training_cost += best_training_loss
							total_training_stat += best_training_stat
							total_validation_stat += best_validation_stat
							total_avg_epoch += avg_epoch
							break
				tf.reset_default_graph()
					
			total_validation_stat = total_validation_stat/float(num_models)
			total_training_cost = total_training_cost/float(num_models)
			total_training_stat = total_training_stat/float(num_models)
			total_avg_epoch = total_avg_epoch/float(num_models)
			
				

			if total_validation_stat > best_model_stat:
				best_model_stat = total_validation_stat
				train_target = total_training_cost
				best_lambda = lamb
				best_drop = drop
				max_epoch = np.round(total_avg_epoch)
					
	return best_num_layers, best_num_nodes, best_lambda, best_drop, int(np.round(max_epoch + 5.1, -1))
