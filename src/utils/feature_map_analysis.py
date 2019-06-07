import sys
import pandas as pd
import numpy as np
import argparse
import utils.graph
from joblib import Parallel, delayed
import multiprocessing
from math import ceil

num_cores = multiprocessing.cpu_count()


#Convert abundance vector into tree matrix
def generate_maps(i, j, theta, lab, max_list, fm_cols, w, ref, data, num_samp):
	id = multiprocessing.Process()._identity
	w_row = w.shape[0]
	w_col = w.shape[1]
	dictionary={}
	loc_list = max_list[i][j].argsort()[::-1]
	for k in range(0, len(loc_list)):
		loc = loc_list[k]
		max_count = max_list[i][j][loc]
		if max_count == 0:
			break
		if max_count >= np.round(num_samp[i] * int(theta)):
			row = int(loc) // int(fm_cols)
			col = loc % fm_cols
			ref_window = ref[row:row + w_row, col:col + w_col]
			count = np.zeros((w_row,w_col))

			for l in range(0,int(num_samp[i])):
				window =data[lab][l, row:row + w_row, col:col + w_col]
				abs_v = (abs(w[:,:,j]) * abs(window)).sum() + 0.00001
				v = (w[:,:,j] * window)
				for m in range(0, v.shape[0]):
					for n in range(0, v.shape[1]):
						count[m,n] += v[m,n] / abs_v

			count = count/num_samp[i]
			for m in range(0, ref_window.shape[0]):
				for n in range(0, ref_window.shape[1]):
					if ref_window[m,n] in dictionary and ref_window[m,n] != "0":
						if dictionary[ref_window[m,n]] < count[m,n]:
							dictionary[ref_window[m,n]] = count[m,n]
					elif ref_window[m,n] != "0":
						dictionary[ref_window[m,n]] = count[m,n]
	return [dictionary]


def get_feature_map_rankings_cnn(x, y, pred, fm, w, b, g, label_set, features):
	fm = np.array(fm)
	w = np.array(w)
	x = np.array(x)
	y = np.array(y)
	#fm = np.squeeze(fm)
	w = np.squeeze(w)
	x = np.squeeze(x)

	ref = g.get_ref()

	ref_val = np.zeros((ref.shape[0], ref.shape[1]))
	num_nodes = g.get_node_count()

	rankings = {}
	scores = {}
	node_names = g.get_dictionary()

	labels = label_set
	otus = features

	num_classes = len(labels)

	for i in labels:
		rankings[i] = {}
		scores[i] = {}
		for j in node_names:
			rankings[i][j] = []
			scores[i][j] = []


	total_num_samp = x.shape[0]
	num_maps = fm.shape[-1]
	w_row = w.shape[0]
	w_col = w.shape[1]

	data = {}
	fm_data = {}

	for i in labels:
		data[i] = []
		fm_data[i] = []

	for i in range(total_num_samp):
		#if y[i] == pred[i]:
		l = y[i]
		data[labels[l]].append(x[i])
		fm_data[labels[l]].append(fm[i])

	num_samp = np.zeros(num_classes)

	for i in range(len(labels)):
		l = labels[i]
		num_samp[i] = int(len(data[l]))
		data[l] = np.array(data[l])
		fm_data[l] = np.array(fm_data[l])

	fm_rows = fm.shape[1]
	fm_cols = fm.shape[2]

	theta1 = 0.05
	theta2 = 0.10
	#Get the top X max indices for each class and each feature map
	max_list = np.zeros((num_classes, num_maps, fm_rows * fm_cols))
	for i in range(len(labels)):
		lab = labels[i]
		for j in range(0, int(num_samp[i])):
			for k in range(0, num_maps):
				maximums = np.argsort(fm_data[lab][j,:,:,k].flatten())[::-1]
				for l in range(0, int(ceil(fm_rows * fm_cols * theta1))):
					if (fm_data[lab][j,:,:,k].flatten()[maximums[l]] > b[k]):
							max_list[i][k][maximums[l]] += 1
					else:
						break
	d = {"OTU":otus,"Max Score":np.zeros(len(otus)), "Cumulative Score":np.zeros(len(otus))}
	df = pd.DataFrame(data = d)
	results = {}


	for i in labels:
		results[i] = df.set_index("OTU")

	for i in range(len(labels)):
		lab = labels[i]
		#For each feature map...
		fm_results = Parallel(n_jobs=num_cores)(delayed(generate_maps)(i, j, 0.1, lab, max_list, fm_cols, w, ref, data, num_samp) for j in range(0, num_maps))
		my_fm_results = np.array(np.take(fm_results,0,1))

		for d in my_fm_results:
			for f in d.keys():
				if f != "0":
					if d[f] > 0 and f in results[lab].index:
						if d[f] > results[lab].loc[f, "Max Score"]:
							results[lab].loc[f, "Max Score"] = d[f]


		'''
		for j in range(0, num_maps):
			#Order the feature map's maximums
			loc_list = max_list[i][j].argsort()[::-1]
			#Store kernel weights
			#For the top X maximums...i
			for k in range(0, len(loc_list)):
				#Find the row and column location and isolate reference window
				loc = loc_list[k]
				max_count = max_list[i][j][loc]
				if max_count == 0:
					break
				if max_count >= int(round(num_samp[i] * theta2)):
					row = int(loc) // int(fm_cols)
					col = loc % fm_cols
					ref_window = ref[row:row + w_row, col:col + w_col]
					count = np.zeros((w_row,w_col))

					#Calculate the proportion of the contribution of each pixel to the convolution with the absolute value of weights
					for l in range(0,int(num_samp[i])):
						window =data[lab][l, row:row + w_row, col:col + w_col]
						abs_v = (abs(w[:,:,j]) * abs(window)).sum() + 0.00001
						v = (w[:,:,j] * window)
						for m in range(0, v.shape[0]):
							for n in range(0, v.shape[1]):
								count[m,n] += v[m,n] / abs_v

					#Divide by number of samples
					count = count/num_samp[i]
		#Print out features with a high enough value
		for m in range(0, w_row):
			for n in range(0, w_col):
				if count[m,n] > 0:
					if ref_window[m,n] in results[lab].index:
						if count[m,n] > results[lab].loc[ref_window[m,n], "Max Score"]:
							results[lab].loc[ref_window[m,n], "Max Score"] = count[m,n]
							results[lab].loc[ref_window[m,n], "Cumulative Score"] += count[m,n]
						else:
							results[lab].loc[ref_window[m,n], "Max Score"] = count[m,n]
							results[lab].loc[ref_window[m,n], "Cumulative Score"] = count[m,n]


	'''
	diff = {}
	for i in labels:
		diff[i] = df.set_index("OTU")
		for j in results[i].index:
		   for k in labels:
			   if i != k:
				   if j in results[k].index:
					   diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"] - results[k].loc[j, "Max Score"]
				   else:
					   diff[i].loc[j, "Max Score"] = results[i].loc[j, "Max Score"]

		rank = diff[i]["Max Score"].rank(ascending=False)

		for j in node_names:
			if j in rank.index:
				rankings[i][j] = rank.loc[j]
				scores[i][j] = diff[i].loc[j,"Max Score"]
			else:
				rankings[i][j] = rank.shape[0] + 1
				scores[i][j] = 0

	score_df = {}
	rank_df = {}
	for i in labels:
		score_df[i] = pd.DataFrame.from_dict(data=scores[i], orient='index', columns=["Score"])
		rank_df[i] = pd.DataFrame.from_dict(data=rankings[i], orient='index', columns=["Rank"])
	return score_df

def get_feature_rankings_mlpnn(weight_list):
	num_layers = len(weight_list)
	cummulative = weight_list[0]
	for w in range(1,num_layers):
		cummulative = np.matmul(cummulative, weight_list[w])
	return cummulative

def get_feature_map_rankings_cnn_2(kernels, weight_list, g, features):
	ref = g.get_ref()

	k_height = int(kernels.shape[0])
	k_width = int(kernels.shape[1])
	num_kernels = kernels.shape[-1]
	num_layers = len(weight_list)

	cummulative = weight_list[0]

	for w in range(1, num_layers):
		cummulative = np.matmul(cummulative, weight_list[w])

	nodes_per_kernel = cummulative.shape[0]/num_kernels
	num_class = cummulative.shape[1]
	out_df = pd.DataFrame(index=features, columns=[str(x) for x in range(0, num_class)])
	out_df = out_df.fillna(0.0)
	for c in range(0,num_class):
		for pos in range(0,cummulative.shape[0]):
			k = int(pos / nodes_per_kernel)
			offset = int(pos % nodes_per_kernel)
			fm_rows = ref.shape[0] - k_height + 1
			fm_cols = ref.shape[1] - k_width + 1

			col_off = int(offset % fm_cols)
			row_off = int(offset / fm_cols)


			window = ref[row_off:row_off+k_height,col_off:col_off+k_width]
			temp_cummulative = np.squeeze(cummulative[pos,c] * kernels[:,:,:,k])
			for x in range(0,window.shape[0]):
				for y in range(0,window.shape[1]):
					feat = window[x,y]
					if feat != 0 and feat != '0':
						score = temp_cummulative[x,y]
						out_df.loc[feat][str(c)] += score

	return out_df.values
