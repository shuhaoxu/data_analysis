# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:03:22 2018

@author: 18810
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load sensor data
#file_prefix = os.path
#print ("current path : {}".format(file_prefix))
filename = "./touch_events.csv"
src_df = pd.read_csv(filename, sep=",")
col    = src_df.columns
#print ("column name of dataframe : {}".format(col))
# extract main sensor data from dataframe
val     = src_df['event'].values
new_val = []
i = 0
for a in val:
    aa = re.split('[\[\]]', a)
    new_val.append([])
    j = 0
    for m in range(len(aa)):
        if len(aa[m]) > 2:
            new_val[i].append([])
            b = aa[m].split()
            for bb in b:
                bb = bb.split(',')
                new_val[i][j].append(float(bb[0]))
            j += 1
    i += 1
new_val = np.array(new_val, dtype=float)
d0, d1, d2 = new_val.shape
# compute mean value of 3rd-dim value
val = np.mean(new_val, axis=2)
threshold = (val.max() - val.min()) / 2
val_mean = np.mean(val, axis=1)
t   = np.arange(1, 236, 2)

# plot value line : all 100-dim data, use val
# =============================================================================
# plt.figure(figsize=(20, 10))
# plt.plot(t, val)
# plt.ylabel("mean value of 8-dim data")
# plt.title("overview_100")
# plt.axis([0, 240, 0, 1.5])
# #plt.show()
# plt.savefig("overview_100.png")
# =============================================================================
# =============================================================================
# # plot value line : mean value of (118, 100) data, use val_mean
# plt.figure(figsize=(20, 10))
# plt.plot(t, val_mean)
# plt.ylabel("mean value of (100, 8) data")
# plt.title("overview_mean")
# plt.axis([0, 240, 0, 1.5])
# #plt.show()
# plt.savefig("overview_mean.png")
# =============================================================================

# generate random noize data
# =============================================================================
# rand_size = 10
# np.random.seed(1)
# rand = np.random.rand(rand_size)
# rand_idx = np.random.randint(0, d0, rand_size)
# #mu, sigma = 0, 0.1
# #rand = np.random.normal(mu, sigma, d0)
# #np.random.shuffle(rand)
# nval = val_mean.copy()
# tmp  = nval[rand_idx]
# tmp  = tmp + rand
# nval[rand_idx] = tmp
# plt.figure(figsize=(20, 10))
# plt.plot(t, val_mean, "g")
# plt.plot(t, nval, "r")
# #plt.plot(t, nval)
# plt.axis([0, 240, 0, 2.1])
# #plt.show()
# #plt.savefig("add_noise.png")
# #plt.savefig("compare.png")
# =============================================================================

# analyse each dim of 118 data
# =============================================================================
# x = np.arange(1, 201, 2)
# #define the figure size and grid layout properties
# import matplotlib.gridspec as gridspec
# cols = 10
# figsize = (30, 30)
# xy_lim  = [0, 210, 0, 2]
# gs = gridspec.GridSpec(val.shape[0] // cols + 1, cols)
# gs.update(hspace=0.4)
# #plot each dim. data
# fig1 = plt.figure(num=1, figsize=figsize)
# ax = []
# for i in range(val.shape[0]):
#     dim = val[i, :]
#     row = (i // cols)
#     col = i % cols
#     ax.append(fig1.add_subplot(gs[row, col]))
#     ax[-1].set_title("dim{}".format(i))
#     ax[-1].plot(x, dim)
#     ax[-1].axis(xy_lim)
# plt.show()
# fig1.savefig("100-dim_show.png")
# =============================================================================
# =============================================================================
# dim1 = val[0, :]
# plt.plot(t, dim1)
# plt.ylabel("dim1 of data")
# plt.title("observe dim1")
# #plt.axis([0, 240, 0, int(dim1.max())+1])
# plt.axis([0, 240, 0, 1])
# =============================================================================

# analyse data of dim6 : segment node
x = np.arange(1, 101, 1)
#dim6 = val[6, :]
dim = val[7, :]
thsh = (dim.max() - dim.min()) / 2
idx = []
min_idx = 0
max_idx = 0
for i in range(1, dim.shape[0]):
    ch_tmp = np.abs(dim[i] - dim[i-1])
    ch_max = np.abs(dim[i] - dim[max_idx])
    if ch_tmp >= thsh:
        min_idx = i - 1
        idx.append(min_idx + 0.5)
        max_idx = i
    if ch_max < 0.05:
        ch_min = np.abs(dim[i] - dim[min_idx])
        if ch_min >= thsh:
            min_idx = max_idx
            idx.append(min_idx + 1.5)
        max_idx = i
#plot data
xy_lim = [0, 110, 0, 2]
plt.figure(figsize=(20,10))
#plt.plot(x, dim6, "g", x, thid, "ro")
plt.plot(x, dim, "g")
plt.vlines(idx, 0, 2, colors="r", linestyles="dashed")
plt.axis(xy_lim)
plt.show()