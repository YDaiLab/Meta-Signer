from scipy.stats import ranksums
import numpy as np
import pandas as pd

def get_wilcoxon_ranked_list(x, y, feature_list, label_set):

	if len(label_set) == 2:
		values = []
		for f in range(len(feature_list)):
			sub_df_0 = x[y==0,f]
			sub_df_1 = x[y==1,f]
			values.append(ranksums(sub_df_0.reshape(-1), sub_df_1.reshape(-1)).pvalue)
	else:
		values = []
		for c in range(len(label_set)):
			sub_values = []
			for f in range(len(feature_list)):
				sub_df_0 = x[y==c,f]
				sub_df_1 = x[y!=c,f]
				sub_values.append(ranksums(sub_df_0.reshape(-1), sub_df_1.reshape(-1)).pvalue)
			values.append(sub_values)
		values = np.amin(np.array(values), axis=0)
	
	wilcox_df = pd.DataFrame(index=feature_list, data=np.array(values).reshape(-1))
	return wilcox_df
