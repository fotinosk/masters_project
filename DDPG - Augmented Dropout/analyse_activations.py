import numpy as np
from numpy import genfromtxt
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import combinations

from numpy.lib.histograms import _hist_bin_fd

cwd = os.getcwd()
path = os.path.join(cwd, 'Visualizer/activations')
os.chdir(path)
activation_files = os.listdir()

big_dict = {i: genfromtxt(i, delimiter=',', dtype=float) for i in activation_files}
arr = np.array(activation_files)
arr = np.reshape(arr, (7, -1))

# l1_actors = arr[:,0]
# fig, ax = plt.subplots(len(l1_actors), len(l1_actors), sharey=True)
# fig.suptitle('Actor Layer 1')
# for i in range(len(l1_actors)):
#     ax[0,i].set_title(l1_actors[i][:2])
#     ax[i,0].set_ylabel(l1_actors[i][:2])

# for i in range(len(l1_actors)):
#     for j in range(len(l1_actors)):
#         diff = big_dict[l1_actors[i]] - big_dict[l1_actors[j]]
#         ar = 0.01 * 100 / len(diff)
#         diff = np.reshape(diff, (-1,1))
#         # ax[i,j].axis('off')
#         ax[i,j].set_yticklabels([])
#         ax[i,j].set_xticklabels([])
#         ax[i,j].matshow(diff, aspect=ar, cmap=cm.bwr)

# plt.show()

def plot(index):
    l1 = arr[:,index]
    fig, ax = plt.subplots(len(l1), len(l1), sharey=True)
    fig.suptitle(l1[0][2:-4].strip())
    for i in range(len(l1)):
        ax[0,i].set_title(l1[i][:2])
        ax[i,0].set_ylabel(l1[i][:2])

    for i in range(len(l1)):
        for j in range(len(l1)):
            diff = big_dict[l1[i]] - big_dict[l1[j]]
            ar = 0.01 * 100 / len(diff)
            diff = np.reshape(diff, (-1,1))
            ax[i,j].set_yticklabels([])
            ax[i,j].set_xticklabels([])
            ax[i,j].matshow(diff, aspect=ar, cmap=cm.bwr)

    plt.show()

# for i in range(arr.shape[1]):
#     plot(i)
