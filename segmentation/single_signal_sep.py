# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:47:41 2018
function : segmentation on single sensor signal data
@author: xushuhao
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data_file = "./touch_events.csv"

# Load appointed signal data : event
def load_data(filename):
    # load raw *.csv data
    src_df = pd.read_csv(filename, sep=",")

    # get column names of src_df
    cols = src_df.columns

    # get 'event' column data from raw data
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
    new_val = np.array(new_val).astype(np.float32)
    new_val = np.mean(new_val, axis=2)

    return new_val

# Segment data by thrift change
def seg_signal(signal):
    # compute threshold for signal
    thsh = (signal.max() - signal.min()) / 2
    
    # initialize idx for segmentation data
    min_idx, max_idx = 0, 0
    idx = []
    
    # search segmentation nodes
    for i in range(1, signal.shape[0]):
        ch_tmp = np.abs(signal[i] - signal[i-1])
        ch_max = np.abs(signal[i] - signal[max_idx])
        if ch_tmp >= thsh:
            min_idx = i - 1
            max_idx = i
            idx.append(min_idx - 0.5)
        if ch_max < 0.05:
            ch_min = np.abs(signal[i] - signal[min_idx])
            if ch_min >= thsh:
                min_idx = max_idx
                idx.append(min_idx + 0.5)
            max_idx = i

    # idx = np.array(idx).astype(int)

    return idx

# Draw subplots for data
def draw_subplots(data, seg_idx):
    # only receive 2-dim data
    d0, d1 = data.shape
    
    # set x for xlabels
    x = np.arange(1, d1+1, 1)
    ymin = 0
    ymax = 2

    # define the figure size and grid layoutproperties
    cols = 10
    figsize = (30, 30)
    xy_lim  = [0, 110, 0, 2]
    gs = gridspec.GridSpec(d0 // cols + 1, cols)
    gs.update(hspace=0.4)
    
    # plot each dim. data
    fig1 = plt.figure(num=1, figsize=figsize)
    ax = []
    for i in range(d0):
        dim = data[i, :]
        idx = np.array(seg_idx[i]).astype(np.float32)
        idx = idx + 1
        row = i // cols
        col = i % cols
        ax.append(fig1.add_subplot(gs[row, col]))
        ax[-1].set_title("dim{}".format(i))
        ax[-1].plot(x, dim, "g")
        ax[-1].vlines(idx, ymin, ymax, colors="r", linestyles="dashed")
        ax[-1].axis(xy_lim)
    fig1.savefig("100-dim_seg_show.png")

# Main function
if __name__ == "__main__":
    # load data from *.csv file
    data = load_data(data_file)
    print ("shape of appointed('event') data : {}".format(data.shape))

    # get segment indexes for data
    seg_idx = []
    for i in range(data.shape[0]):
        tmp_data = data[i, :]
        tmp_idx  = seg_signal(tmp_data)
        seg_idx.append(tmp_idx)
#    seg_idx = np.array(seg_idx).astype(int)

    # draw subplots for data
    draw_subplots(data, seg_idx)
