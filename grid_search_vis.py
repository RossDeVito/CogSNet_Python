import os

import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

if __name__ == "__main__":
	results_file = "mean_df_3.csv"

	results_vis_dir = os.path.join('results', 'vis', results_file.split('.')[0])

	if not os.path.exists(results_vis_dir):
		os.makedirs(results_vis_dir)

	res = pd.read_csv(os.path.join('results', results_file))

	# L value
	sns.lineplot(x='L', y='jaccard_sim', hue='forget_func', ci=95, data=res)
	plt.gca().xaxis.set_major_locator(MultipleLocator(25))
	plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
	plt.grid(True, 'minor', linewidth=1)
	plt.grid(True, 'major', linewidth=2)
	plt.legend(loc='lower right')
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'L'), dpi=150)
	plt.close()

	# mu value
	sns.lineplot(x="mu", y="jaccard_sim", hue='forget_func', data=res)
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'mu'), dpi=150)
	plt.close()

	# theta value
	sns.lineplot(x="theta", y="jaccard_sim", hue='forget_func', data=res)
	plt.tight_layout()
	plt.savefig(os.path.join(results_vis_dir, 'theta'), dpi=150)
	plt.close()

	# # L based facet grid
	# g = sns.FacetGrid(res, col="mu", row="theta", hue="forget_func")
	# g = g.map(plt.scatter, "jaccard_sim", "L").add_legend()
	# g.savefig(os.path.join(results_vis_dir, 'L_grid'), dpi=150)
 
	# # L based facet grid
	# g = sns.FacetGrid(res, col="mu", row="theta", hue="forget_func")
	# g = g.map(plt.scatter, "jaccard_sim", "L").add_legend()
	# g.savefig(os.path.join(results_vis_dir, 'L_grid'), dpi=150)

	# # mu based facet grid
	# g = sns.FacetGrid(res, col="theta", hue="L")
	# g = g.map(plt.scatter, "jaccard_sim", "mu").add_legend()
	# g.savefig(os.path.join(results_vis_dir, 'mu_grid'), dpi=150)

	# # theta based facet grid
	# g = sns.FacetGrid(res, col="mu", hue="L")
	# g = g.map(plt.scatter, "jaccard_sim", "theta").add_legend()
	# g.savefig(os.path.join(results_vis_dir, 'theta_grid'), dpi=150)


