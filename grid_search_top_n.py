import os 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

if __name__ == "__main__":
	results_file = "mean_df_6.csv"
	n = 100

	results_vis_dir = os.path.join('results', 'vis', results_file.split('.')[0])

	if not os.path.exists(results_vis_dir):
		os.makedirs(results_vis_dir)

	res = pd.read_csv(os.path.join('results', results_file))

	top_n_res = res.iloc[np.argsort(res.jaccard_sim)[-n:]]

	# L
	sns.countplot(x="L", data=top_n_res)
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'top_n_L'), dpi=150)
	plt.close()

	# mu
	sns.countplot(x="mu", data=top_n_res)
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'top_n_mu'), dpi=150)
	plt.close()

	# theta
	sns.countplot(x="theta", data=top_n_res)
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'top_n_theta'), dpi=150)
	plt.close()

	# forget fn
	sns.countplot(x="forget_func", data=top_n_res)
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'top_n_ff'), dpi=150)
	plt.close()

	# L dist
	sns.distplot(top_n_res.L.unique())
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'top_n_L_dist'), dpi=150)
	plt.close()
