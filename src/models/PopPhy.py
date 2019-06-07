import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve
from utils.popphy_io import get_stat, get_stat_dict
from utils.feature_map_analysis import get_feature_map_rankings_cnn, get_feature_map_rankings_cnn_2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		

def PopPhyCNN(num_row, num_col, num_class, num_kernel, kernel_height, kernel_width, num_fc_nodes, lamb=0.1, drop=0.1):

	X = tf.placeholder("float", [None,  num_row, num_col, 1], name="X")
	Y = tf.placeholder("float", [None], name="Y")
	Noise = tf.placeholder("float", [None, num_row, num_col, 1], name="Noise")
	CW = tf.placeholder("float", [None], name="CW")
	training = tf.placeholder(tf.bool)
	batch_size = tf.placeholder(tf.int32)
	lab = tf.cast(Y, dtype=tf.int32)
	labels_oh = tf.one_hot(lab, depth=num_class)

	regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
	num_kernels = num_kernel	

	with tf.variable_scope("CNN", reuse=False):
	
		X = tf.add(X, Noise)
		X = tf.layers.dropout(X, rate=0.0, training=training)
		conv = tf.layers.conv2d(X, num_kernels, (kernel_height, kernel_width), name="Conv", activation=None,
			kernel_regularizer=regularizer, bias_regularizer=regularizer, kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
		conv = tf.nn.relu(conv)
		conv_pool = tf.layers.max_pooling2d(conv, (2,2), (2,2))

	with tf.variable_scope("FC"):
		fc = tf.contrib.layers.flatten(conv)
		
		fc = tf.layers.dense(fc, num_fc_nodes, activation=tf.nn.relu, kernel_regularizer=regularizer, bias_regularizer=regularizer, name="Layer1")
		fc = tf.layers.dropout(fc, rate=drop, training=training)
	
	with tf.variable_scope("Output"):						
		out = tf.layers.dense(fc, num_class, kernel_regularizer=regularizer, bias_regularizer=regularizer, name="Out")

	prob = tf.nn.softmax(out, axis=1)

	with tf.name_scope("Loss"):
		ce_loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_oh, logits=out, weights=CW)
		losses = ce_loss + tf.losses.get_regularization_loss()
		cost = tf.reduce_sum(losses)

	return {'ce_cost': ce_loss, 'cost': cost, 'x':X, 'y':Y, 'cw':CW, 'prob':prob, 'pred':prob[:,1], 'training':training, 'noise':Noise, 'batch_size':batch_size, 'fm':conv}


def train(train, test, config, g, params, metric, label_set, features, seed=42):

	learning_rate = float(config.get('PopPhy', 'LearningRate'))
	batch_size = int(config.get('PopPhy', 'BatchSize'))

	num_kernel = params["Num_Kernel"]
	kernel_width = params["Kernel_Size"]
	kernel_height = params["Kernel_Size"]
	num_nodes = params["Num_Nodes"]
	lamb = params["L2_Lambda"]
	drop = params["Dropout_Rate"]
	max_epoch = params["Max_Epoch"]
	

	model_num = 1	
	test_stat = 0
	feature_scores = {}
	feature_rankings = {}
	
	train_x, train_y = train
	test_x, test_y = test

	num_class = len(label_set)
	
	train_x = np.array(train_x)
	test_x = np.array(test_x)
	
	rows = train_x.shape[1]
	cols = train_x.shape[2]
				
	test_preds = []
	test_preds = np.array(test_preds)
	test_probs = []
	test_probs = np.array(test_probs)
	
	train_x = np.array(train_x, dtype=np.float32)

	c_prob = [1] * num_class
	for l in np.unique(train_y):
		c_prob[int(l)] = float( float(len(train_y))/ (2.0 * float((np.sum(train_y == l)))))
		
	train_weights = []
	test_weights = []
					
	for i in train_y:
		train_weights.append(c_prob[int(i)])

	for i in test_y:
		test_weights.append(c_prob[int(i)])


	train_x = np.expand_dims(train_x, -1)
	test_x = np.expand_dims(test_x, -1)

	mask = np.array(g.get_mask()).reshape(1,rows,cols,1)
					
	train_weights = np.array(train_weights)
	test_weights = np.array(test_weights)
	
	final_test_probs = []
	
	train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
	train_dataset = train_dataset.shuffle(100000).batch(batch_size)
		
	test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y, test_weights))
	test_dataset = test_dataset.batch(1024)
								
	train_iterator = train_dataset.make_initializable_iterator()
	next_train_element = train_iterator.get_next()

	test_iterator = test_dataset.make_initializable_iterator()
	next_test_element = test_iterator.get_next()
						
	model = PopPhyCNN(rows, cols, num_class, num_kernel, kernel_height, kernel_width, num_nodes, lamb, drop=drop)

	with tf.name_scope("Train"):
		optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		i=0
		num_training_samples = train_x.shape[0]

		while True:
			i += 1
			training_loss = 0
			test_loss = 0
			num_training_batches = 0
			num_test_batches = 0
								
			sess.run(train_iterator.initializer)
							
			training_y_list = []
			training_pred_list = []
			training_prob_list = []
					
			while True:
				try:
					batch_x, batch_y, batch_cw = sess.run(next_train_element)
					size = batch_x.shape[0]
					noise = np.multiply(np.random.normal(0, 0.1, list(batch_x.shape)), mask)
					_, l, prob, pred = sess.run([optimizer, model['ce_cost'], model['prob'], model['pred']], 
						feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw, 
						model['training']:True, model['noise']:noise, model['batch_size']:size})
					training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
					training_pred_list = list(training_pred_list) + list(pred)
					training_loss += l
					num_training_batches += 1
					if len(training_prob_list) == 0:
						training_prob_list = prob
					else:
						training_prob_list = np.concatenate((training_prob_list, prob), axis=0)

				except tf.errors.OutOfRangeError:
					training_stat = get_stat(training_y_list, training_prob_list, metric)
					training_loss = training_loss/num_training_batches
					break

										
			sess.run(test_iterator.initializer)

			test_y_list = []
			test_pred_list = []
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
					test_pred_list = list(test_pred_list) + list(pred)
					test_loss += l
					num_test_batches += 1
					if len(test_prob_list) == 0:
						test_prob_list = prob
					else:
						test_prob_list = np.stack((test_prob_list, prob))
								
				except tf.errors.OutOfRangeError:
					test_stat = get_stat(test_y_list, test_prob_list, metric)
					test_loss = test_loss/num_test_batches
					break
																	
			if i == max_epoch:
				noise = np.zeros(train_x.shape)
				if model_num == 1:
					scores = []
					fm, prob= sess.run([model['fm'], model['prob']], feed_dict={model['x']:train_x, model['y']: train_y.reshape(-1), model['cw']:train_weights,
						model['training']:False, model['noise']:noise, model['batch_size']:train_x.shape[0]})
					with tf.variable_scope('CNN', reuse=True):
						w = tf.get_variable('Conv/kernel').eval()
						b = tf.get_variable('Conv/bias').eval()
						eval_train = train_x
						eval_y = train_y
						eval_l = np.argmax(prob, axis=1)
					out_scores = get_feature_map_rankings_cnn(eval_train, eval_y, eval_l, fm, w, b, g, label_set, features)					

					for l in label_set:
						scores.append(out_scores[l].loc[features].values)

					scores = np.array(scores)
					scores = np.transpose(scores,(2,1,0))
					#scores = []
					#weight_list = []
					#with tf.variable_scope('CNN', reuse=True):
					#	kernels = tf.get_variable('Conv/kernel').eval()
					#with tf.variable_scope("FC", reuse=True):	
					#	w = tf.get_variable('Layer1/kernel').eval()
					#	weight_list.append(w)
					#with tf.variable_scope("Output", reuse=True):	
					#	w = tf.get_variable('Out/kernel').eval()
					#	weight_list.append(w)
					#eval_train = train_x
					#eval_y = train_y
					#eval_l = np.argmax(prob, axis=1)
					#scores.append(get_feature_map_rankings_cnn_2(kernels, weight_list, g, features))

				test_preds = test_pred_list
				test_probs = test_prob_list
				break	
					
	tf.reset_default_graph()
		
	
	test_stat_dict = get_stat_dict(test_y, test_probs)
	if num_class == 2:
		fpr, tpr, thresh = roc_curve(test_y, test_probs[:,1])
	else:
		fpr, tpr, thresh = 0, 0, 0
		
	#feature_scores_out = {}
	#for l in label_set:
	#	feature_scores_out[l] = feature_scores[l].mean(axis=1).to_frame()
	feature_scores_out = np.mean(scores, axis=0) 
	return test_stat_dict, tpr, fpr, thresh, feature_scores_out, test_probs
	
def tune(train, config, g, seed=42):

	num_models = int(config.get('Evaluation', 'NumberValidationModels'))
	learning_rate = float(config.get('PopPhy', 'LearningRate'))
	batch_size = int(config.get('PopPhy', 'BatchSize'))
	max_patience = int(config.get('PopPhy', 'Patience'))

	x, y = train
	model_run = 0
	num_class = len(np.unique(y))

	if num_class == 2:
		metric = "AUC"
	else:
		metric = "MCC"
	
	x = np.array(x)
	
	rows = x.shape[1]
	cols = x.shape[2]
	
	test_stat = 0
	train_target = 0
	best_model_stat = 0
	mean_cost_target = 0
	best_num_kernel = 8
	best_kernel_width = 5
	best_kernel_height = 5
	best_num_nodes = 32
	best_lambda = 0.001
	best_drop = 0.1
	max_epoch = 0	
	
	skf = StratifiedKFold(n_splits=num_models, random_state=42, shuffle=True)

	
	kernel_list = [16, 32, 64]
	kernel_size_list = [3, 5, 7]
		
	for num_kernel in kernel_list:
		for kernel_size in kernel_size_list:
			total_training_cost = 0
			total_validation_stat = 0
			total_training_stat = 0
			model_num = 0
			kernel_width = kernel_size
			kernel_height = kernel_size
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
				test_weights = []
				full_weights = []
					
				for i in train_y:
					train_weights.append(c_prob[int(i)])

				for i in validation_y:
					validation_weights.append(c_prob[int(i)])
						
				for i in y:
					full_weights.append(c_prob[int(i)])
		
					
				train_x = np.array(train_x).reshape((-1,rows,cols,1))
				validation_x = np.array(validation_x).reshape((-1,rows,cols,1))
					
				train_var = np.var(train_x, axis=0)
				mask = np.array(g.get_mask()).reshape(1,rows,cols,1)
					
				train_weights = np.array(train_weights)
				validation_weights = np.array(validation_weights)
				
				train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
				train_dataset = train_dataset.shuffle(100000).batch(batch_size)

				valid_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y, validation_weights))
				valid_dataset = valid_dataset.shuffle(100000).batch(1024)

				full_dataset = tf.data.Dataset.from_tensor_slices((train_x.reshape(-1, rows, cols, 1), train_y, train_weights))
				full_dataset = full_dataset.shuffle(100000).batch(1024)
					
				train_iterator = train_dataset.make_initializable_iterator()
				next_train_element = train_iterator.get_next()

				valid_iterator = valid_dataset.make_initializable_iterator()
				next_valid_element = valid_iterator.get_next()

				full_iterator = full_dataset.make_initializable_iterator()
				next_full_element = full_iterator.get_next()
					
				model = PopPhyCNN(rows, cols, num_class, num_kernel, kernel_height, kernel_width, best_num_nodes, lamb=best_lambda, drop=best_drop)

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
					num_training_samples = train_x.shape[0]
					num_validation_samples = validation_x.shape[0]
					avg_epoch = 0
						
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
						training_pred_list = []
						training_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_train_element)
								size = batch_x.shape[0]
								noise = np.multiply(np.random.normal(0, 0.1, list(batch_x.shape)), mask)
								_, l, pred, prob = sess.run([optimizer, model['ce_cost'], model['pred'], model['prob']], 
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), 
									model['cw']:batch_cw, model['training']:True, model['noise']:noise, model['batch_size']:size})
								training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
								training_pred_list = list(training_pred_list) + list(pred)
								if len(training_prob_list) == 0:
									training_prob_list = prob
								else:
									training_prob_list = np.concatenate((training_prob_list, prob), axis=0)
								training_loss += l
								num_training_batches += 1

							except tf.errors.OutOfRangeError:	
								training_stat = get_stat_dict(training_y_list, training_prob_list)[metric]
								training_loss = training_loss/num_training_batches
								break

						sess.run(valid_iterator.initializer)
							
						validation_y_list = []
						validation_pred_list = []
						validation_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_valid_element)
								noise = np.zeros(batch_x.shape)
								size = batch_x.shape[0]
								l, pred, y_out, prob = sess.run([model['ce_cost'], model['pred'], model['y'], model['prob']],
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw,
									model['training']:False, model['noise']:noise, model['batch_size']:size})
								validation_y_list = list(validation_y_list) + list(y_out)
								validation_pred_list = list(validation_pred_list) + list(pred)
								if len(validation_prob_list) == 0:
									validation_prob_list = prob
								else:
									validation_prob_list = np.concatenate((validation_prob_list, prob), axis=0)
								validation_loss += l
								num_validation_batches += 1

							except tf.errors.OutOfRangeError:
								patience -= 1
								validation_loss = validation_loss/num_validation_batches
								validation_stat = get_stat_dict(validation_y_list, validation_prob_list)[metric]
								if validation_loss < best_validation_loss or best_validation_stat < validation_stat:
									if best_validation_stat < validation_stat:
										avg_epoch = i
										best_validation_stat = validation_stat
									if validation_loss < best_validation_loss:
										best_training_loss = training_loss
										best_validation_loss = validation_loss
									patience = max_patience
								break
									
						if i > 10 and (training_stat == 0.5 and validation_stat == 0.5):
							i=0
							sess.run(init)
							best_validation_loss = 10000
							best_training_loss = 0
							best_validation_stat = 0
							best_training_stat = 0
							patience=max_patience
			
						if patience == 0 or i > 1000:
							total_training_cost += training_loss
							total_validation_stat += best_validation_stat
							total_training_stat += best_training_stat
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
				best_kernel_height = kernel_height
				best_kernel_width = kernel_width
				best_num_kernel = num_kernel
				max_epoch = np.round(total_avg_epoch)



	num_nodes_list = [8,32,128]
	
	for num_nodes in num_nodes_list:
		total_training_cost = 0
		total_validation_stat = 0
		total_training_stat = 0
		total_avg_epoch = 0

		model_num = 0
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
			test_weights = []
			full_weights = []
					
			for i in train_y:
				train_weights.append(c_prob[int(i)])

			for i in validation_y:
				validation_weights.append(c_prob[int(i)])
						
			for i in y:
				full_weights.append(c_prob[int(i)])
		
					
			train_x = np.array(train_x).reshape((-1,rows,cols,1))
			validation_x = np.array(validation_x).reshape((-1,rows,cols,1))
					
			train_var = np.var(train_x, axis=0)
			mask = np.array(g.get_mask()).reshape(1,rows,cols,1)
					
			train_weights = np.array(train_weights)
			validation_weights = np.array(validation_weights)
				
			train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
			train_dataset = train_dataset.shuffle(100000).batch(batch_size)

			valid_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y, validation_weights))
			valid_dataset = valid_dataset.shuffle(100000).batch(1024)

			full_dataset = tf.data.Dataset.from_tensor_slices((train_x.reshape(-1, rows, cols, 1), train_y, train_weights))
			full_dataset = full_dataset.shuffle(100000).batch(1024)
					
			train_iterator = train_dataset.make_initializable_iterator()
			next_train_element = train_iterator.get_next()

			valid_iterator = valid_dataset.make_initializable_iterator()
			next_valid_element = valid_iterator.get_next()

			full_iterator = full_dataset.make_initializable_iterator()
			next_full_element = full_iterator.get_next()
								
			model = PopPhyCNN(rows, cols, num_class, best_num_kernel, best_kernel_height, best_kernel_width,  num_nodes, lamb=best_lambda, drop=best_drop)

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
				num_training_samples = train_x.shape[0]
				num_validation_samples = validation_x.shape[0]
				avg_epoch = 0

				while True:
					i += 1
					training_loss = 0
					validation_loss = 0
					num_training_batches = 0
					num_validation_batches = 0
					num_test_batches = 0
						
					sess.run(train_iterator.initializer)
							
					training_y_list = []
					training_pred_list = []
					training_prob_list = []
					while True:
						try:
							batch_x, batch_y, batch_cw = sess.run(next_train_element)
							size = batch_x.shape[0]
							noise = np.multiply(np.random.normal(0, 0.1, list(batch_x.shape)), mask)
							_, l, pred, prob = sess.run([optimizer, model['ce_cost'], model['pred'], model['prob']], 
								feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), 
								model['cw']:batch_cw, model['training']:True, model['noise']:noise, model['batch_size']:size})
							training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
							training_pred_list = list(training_pred_list) + list(pred)
							if len(training_prob_list) == 0:
								training_prob_list = prob
							else:
								training_prob_list = np.concatenate((training_prob_list, prob), axis=0)
							training_loss += l
							num_training_batches += 1

						except tf.errors.OutOfRangeError:
							training_stat = get_stat_dict(training_y_list, training_prob_list)[metric]
							training_loss = training_loss/num_training_batches
							break

					sess.run(valid_iterator.initializer)
							
					validation_y_list = []
					validation_pred_list = []
					validation_prob_list = []
					while True:
						try:
							batch_x, batch_y, batch_cw = sess.run(next_valid_element)
							noise = np.zeros(batch_x.shape)
							size = batch_x.shape[0]
							l, pred, y_out, prob = sess.run([model['ce_cost'], model['pred'], model['y'], model['prob']],
								feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw,
								model['training']:False, model['noise']:noise, model['batch_size']:size})
							validation_y_list = list(validation_y_list) + list(y_out)
							validation_pred_list = list(validation_pred_list) + list(pred)
							if len(validation_prob_list) == 0:
								validation_prob_list = prob
							else:
								validation_prob_list = np.concatenate((validation_prob_list, prob), axis=0)
							validation_loss += l
							num_validation_batches += 1

						except tf.errors.OutOfRangeError:
							patience -= 1
							validation_loss = validation_loss/num_validation_batches
							validation_stat = get_stat(validation_y_list, validation_prob_list, metric)
							if validation_loss < best_validation_loss or best_validation_stat < validation_stat:
								if best_validation_stat < validation_stat:
									best_validation_stat = validation_stat
									avg_epoch = i				
								if validation_loss < best_validation_loss:
									best_training_loss = training_loss
									best_validation_loss = validation_loss
								patience = max_patience
							break

					if i > 10 and (training_stat == 0.5 and validation_stat == 0.5):
						i=0
						sess.run(init)
						best_validation_loss = 10000
						best_training_loss = 0
						best_validation_stat = 0
						best_training_stat = 0
						patience = max_patience
				
					if patience == 0 or i > 1000:
						total_training_cost += training_loss
						total_validation_stat += best_validation_stat
						total_training_stat += best_training_stat
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
			best_num_nodes = num_nodes
			max_epoch = np.round(total_avg_epoch)				
				
				
	lambda_list = [0.01,  0.001]
	drop_list = [0.25, 0.5]
	
	for lamb in lambda_list:
		for drop in drop_list:
			total_training_cost = 0
			total_validation_stat = 0
			total_training_stat = 0
			total_avg_epoch = 0
			model_num = 0
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
				test_weights = []
				full_weights = []
					
				for i in train_y:
					train_weights.append(c_prob[int(i)])

				for i in validation_y:
					validation_weights.append(c_prob[int(i)])
						
				for i in y:
					full_weights.append(c_prob[int(i)])
		
					
				train_x = np.array(train_x).reshape((-1,rows,cols,1))
				validation_x = np.array(validation_x).reshape((-1,rows,cols,1))
					
				train_var = np.var(train_x, axis=0)
				mask = np.array(g.get_mask()).reshape(1,rows,cols,1)
					
				train_weights = np.array(train_weights)
				validation_weights = np.array(validation_weights)
				
				train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_weights))
				train_dataset = train_dataset.shuffle(100000).batch(batch_size)

				valid_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y, validation_weights))
				valid_dataset = valid_dataset.shuffle(100000).batch(1024)

				full_dataset = tf.data.Dataset.from_tensor_slices((train_x.reshape(-1, rows, cols, 1), train_y, train_weights))
				full_dataset = full_dataset.shuffle(100000).batch(1024)
					
				train_iterator = train_dataset.make_initializable_iterator()
				next_train_element = train_iterator.get_next()

				valid_iterator = valid_dataset.make_initializable_iterator()
				next_valid_element = valid_iterator.get_next()

				full_iterator = full_dataset.make_initializable_iterator()
				next_full_element = full_iterator.get_next()
								
				model = PopPhyCNN(rows, cols, num_class, best_num_kernel, best_kernel_height, best_kernel_width, best_num_nodes, lamb=lamb, drop=drop)

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
					num_training_samples = train_x.shape[0]
					num_validation_samples = validation_x.shape[0]
					avg_epoch = 0			
					while True:
						i += 1
						training_loss = 0
						validation_loss = 0
						num_training_batches = 0
						num_validation_batches = 0
						num_test_batches = 0
						
						sess.run(train_iterator.initializer)
							
						training_y_list = []
						training_pred_list = []
						training_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_train_element)
								size = batch_x.shape[0]
								noise = np.multiply(np.random.normal(0, 0.1, list(batch_x.shape)), mask)
								_, l, pred, prob = sess.run([optimizer, model['ce_cost'], model['pred'], model['prob']], 
									feed_dict={model['x']: batch_x,model['y']: batch_y.reshape(-1), model['cw']:batch_cw, 
									model['training']:True, model['noise']:noise, model['batch_size']:size})
								training_y_list = list(training_y_list) + list(batch_y.reshape(-1))
								training_pred_list = list(training_pred_list) + list(pred)
								if len(training_prob_list) == 0:
									training_prob_list = prob
								else:
									training_prob_list = np.concatenate((training_prob_list, prob), axis=0)
								training_loss += l
								num_training_batches += 1

							except tf.errors.OutOfRangeError:
								training_stat = get_stat_dict(training_y_list, training_prob_list)[metric]
								training_loss = training_loss/num_training_batches
								break

						sess.run(valid_iterator.initializer)
							
						validation_y_list = []
						validation_pred_list = []
						validation_prob_list = []
						while True:
							try:
								batch_x, batch_y, batch_cw = sess.run(next_valid_element)
								noise = np.zeros(batch_x.shape)
								size = batch_x.shape[0]
								l, pred, y_out, prob = sess.run([model['ce_cost'], model['pred'], model['y'], model['prob']],
									feed_dict={model['x']: batch_x, model['y']: batch_y.reshape(-1), model['cw']:batch_cw,
									model['training']:False, model['noise']:noise, model['batch_size']:size})
								validation_y_list = list(validation_y_list) + list(y_out)
								validation_pred_list = list(validation_pred_list) + list(pred)
								if len(validation_prob_list) == 0:
									validation_prob_list = prob
								else:
									validation_prob_list = np.concatenate((validation_prob_list, prob), axis=0)
								validation_loss += l
								num_validation_batches += 1

							except tf.errors.OutOfRangeError:
								patience -= 1
								validation_loss = validation_loss/num_validation_batches
								validation_stat = get_stat_dict(validation_y_list, validation_prob_list)[metric]
								if validation_loss < best_validation_loss or best_validation_stat < validation_stat:
									if best_validation_stat < validation_stat:
										avg_epoch = i
										best_validation_stat = validation_stat
									if validation_loss < best_validation_loss:
										best_training_loss = training_loss
										best_validation_loss = validation_loss
									patience = max_patience
								break
								
						if i > 10 and (training_stat == 0.5 and validation_stat == 0.5):
							i=0
							sess.run(init)
							best_validation_loss = 10000
							best_training_loss = 0
							best_validation_stat = 0
							best_training_stat = 0
							patience=max_patience
				
						if patience == 0 or i > 1000:
							total_training_cost += training_loss
							total_validation_stat += best_validation_stat
							total_training_stat += best_training_stat
							total_avg_epoch += avg_epoch
							break
				tf.reset_default_graph()
					
			total_validation_stat = total_validation_stat/float(num_models)
			total_training_cost = total_training_cost/float(num_models)
			total_training_stat = total_training_stat/float(num_models)
			total_avg_epoch= total_avg_epoch/float(num_models)
				
			
			if total_validation_stat > best_model_stat:
				best_model_stat = total_validation_stat
				train_target = total_training_cost
				best_lambda = lamb
				best_drop = drop
				max_epoch = np.round(total_avg_epoch)

	return best_num_kernel, best_kernel_width, best_num_nodes, best_lambda, best_drop, int(np.round(max_epoch + 5.1, -1))
