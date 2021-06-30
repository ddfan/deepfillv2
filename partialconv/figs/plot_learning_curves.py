import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import genfromtxt

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

directory = 'learning_curves'

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

print(onlyfiles)

epoch_train = {}
epoch_val = {}
loss_train = {}
loss_val = {}
names = []
for file in onlyfiles:
	file_split = file.replace('.','-').replace('_','-').split("-")
	if "total" in file_split:
		continue
	name = file_split[-2]
	data = np.genfromtxt(directory + "/" + file, delimiter=',')
	if "train" in file_split:
		epoch_train[name] = data[1:,1]
		loss_train[name] = data[1:,2]
	elif "val" in file_split:
		epoch_val[name] = data[1:,1]
		loss_val[name] = data[1:,2]

	if name not in names:
		names.append(name)

print(names)

f, ax = plt.subplots(nrows=1, ncols=len(names), figsize=(8,2.5))

for i in range(len(names)):
	ax[i].plot(epoch_train[names[i]], loss_train[names[i]])
	ax[i].plot(epoch_val[names[i]], loss_val[names[i]])
	ax[i].legend(['train', 'val'])
	ax[i].set_xlabel("Epoch")
	ax[i].set_title(names[i])

plt.tight_layout()

plt.savefig("learning_curves.pdf", bbox_inches="tight")

plt.show()