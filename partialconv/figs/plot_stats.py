import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import genfromtxt

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# directory = 'compare_models'
# directory = 'in_dist_comp'
directory = 'subway_ground_outdist'

onlyfiles = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

print(onlyfiles)

fig = plt.figure(figsize=(4, 4), dpi=150)

alpha_stats_val = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

colors = plt.cm.plasma(np.linspace(0,1,len(onlyfiles))) 
labels = ['LM','TB','SO','SA','TA','LC']

# colors = plt.cm.brg(np.linspace(1,0,len(onlyfiles))) 
# labels = ['CVaR-Learning', 'L1 Loss', 'Handcrafted']

for i,file in enumerate(onlyfiles):
    data = np.load(file, allow_pickle=True)

    alpha_implied = data.item().get("alpha_implied")
    r2_var = data.item().get("r2_var")
    r2_cvar = data.item().get("r2_cvar")

    # make statistics plot
    boxplot_width = 0.03

    plt.subplot(311)
    if len(alpha_implied.shape) == 2:
        plt.boxplot(alpha_implied, positions=alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'r', 'markeredgecolor': 'r', 'marker': '.', 'markersize': 1})
    else:
        plt.plot(alpha_stats_val, alpha_implied, color=colors[i], marker=".", alpha=0.8)

    plt.ylabel(r'$Implied ~\alpha$')
    # plt.title(directory)
    plt.xlim((min(alpha_stats_val) - boxplot_width, max(alpha_stats_val) + boxplot_width))
    plt.ylim((-0.1, 1.1))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    plt.subplot(312)
    if len(r2_var.shape) == 2:
        plt.boxplot(r2_var, positions=alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'r', 'markeredgecolor': 'r', 'marker': '.', 'markersize': 1})
    else:
        plt.plot(alpha_stats_val, r2_var, color=colors[i], marker=".", alpha=0.8)
        
    plt.ylabel(r"$VaR~R^2$")
    plt.xlim((min(alpha_stats_val) - boxplot_width, max(alpha_stats_val) + boxplot_width))
    plt.ylim((0, 1.0))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    plt.subplot(313)
    if len(r2_cvar.shape) == 2:
        plt.boxplot(r2_cvar, positions=alpha_stats_val, widths=boxplot_width,
            flierprops={'markerfacecolor': 'r', 'markeredgecolor': 'r', 'marker': '.', 'markersize': 1})
    else:
        plt.plot(alpha_stats_val, r2_cvar, color=colors[i], marker=".", alpha=0.8)

    plt.ylabel(r"$CVaR ~ R^2$")
    plt.xlabel(r"$\alpha$")
    plt.xlim((min(alpha_stats_val) - boxplot_width, max(alpha_stats_val) + boxplot_width))
    plt.ylim((0, 1.0))
    plt.gca().set_xticks(alpha_stats_val)
    plt.gca().set_xticklabels([str(alph).lstrip('0') for alph in alpha_stats_val])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

# Shrink current axis by 20%
plt.subplot(311)
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width, 0.8 * box.height])
plt.subplot(312)
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width, 0.8 * box.height])
plt.subplot(313)
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width, 0.8 * box.height])

plt.subplot(311)
# plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 1.45),
          # ncol=len(onlyfiles), prop={'size': 8})
plt.legend(labels, loc='best',
          ncol=int(len(onlyfiles)/2), prop={'size': 7})
plt.savefig(directory + ".pdf", bbox_inches="tight")

plt.show()