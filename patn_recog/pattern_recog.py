#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from read_data import *

data_file = "/home/xsh/download/Sensor_Analysis/touch_events.csv"
#data_file = "./touch_events.csv"

# Analyse 7th dimension data
def dim7_data_analysis(data):
    # Get dimension of raw data
    sample_num, time, sensor_num = data.shape
    print ("sample number : {}, sensor number : {}, time length : {}".format(sample_num, sensor_num, time))
    
    # Analyse 7-th sample
    data = data[6, :, :]
    data_mean = np.mean(data, axis=1)
    print ("shape of 7-th sample : {}".format(data.shape))
    print ("shape of averaged data : {}".format(data_mean.shape))
    
    # Set time length
    t = np.arange(1, 101, 1)
    
    # Compute distance on each sensor
    err = np.zeros((sensor_num, time))
    dist_sep = np.zeros((sensor_num)).reshape(1,sensor_num).ravel()
    print ("shape of err : {}, dist_sep : {}".format(err.shape, dist_sep.shape))
    for j in range(sensor_num):
        for i in range(time):
            err[j,i] = data[i,j] - data_mean[i]
            dist_sep[j] += np.square(err[j,i])
        dist_sep[j] = np.sqrt(dist_sep[j])
    dist_idx  = np.argsort(dist_sep)
    sim_sep  = 1 / dist_sep
    print ("distance between each sensor and mean value :\n{}".format(dist_sep))
    print ("similarity between each sensor and mean value :\n{}".format(sim_sep))
    print ("sort of distance of each sensor : {}".format(dist_idx))
    dist_all = dist_sep.sum()
    sim_all  = 1 / dist_all
    print ("distance between all sensors and mean value : {}".format(dist_all))
    print ("similarity between all sensors and mean value : {}".format(sim_all))

# Draw subplots for data
def draw_subplots_bak(data, seg_idx):
    # only receive 2-dim data
    d0, d1 = data.shape
    
    # set x for xlabels
    x = np.arange(1, d1+1, 1)
    ymin = 0
    ymax = 2

    # define the figure size and grid layout properties
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
#    fig1.savefig("100-dim_seg_show.png")

# Draw subplots for data
def draw_subplots(data, pattern):
    # set x for x labels
    x = np.arange(1, 101, 1)
    ymin = 0
    ymax = 2

    # define the figure size
    cols = 4
    figsize = (30, 30)
    xy_lim  = [0, 110, 0, 2]
    
    for i in range(3):
        # get appointed kind of data and pattern
        test = data[i]
        patn = pattern[i]
        d0, d1, d2 = test.shape
        
        # define grid layout properties
        gs = gridspec.GridSpec(d2 // cols + 1, cols)
        gs.update(hspace=0.4)
        
        # plot each dim. data
        for j in range(d0):
            fig1 = plt.figure(num=1, figsize=figsize)
            ax = []
            sample = test[j]
            for m in range(d2):
                dim = sample[:, m]
                row = m //cols
                col = m % cols
                ax.append(fig1.add_subplot(gs[row, col]))
                ax[-1].set_title("dim{}".format(m))
                ax[-1].plot(x, dim, "b")
                ax[-1].plot(x, patn, "g")
                ax[-1].axis(xy_lim)
            #if i == 0:
            #    fig1.savefig("./out_pic/b/sample{}.png".format(j))
            #elif i == 1:
            #    fig1.savefig("./out_pic/k/sample{}.png".format(j))
            #else:
            #    fig1.savefig("./out_pic/r/sample{}.png".format(j))
            plt.close()
    
    

# Main function
if __name__ == "__main__":
    # Load raw data
    data = load_data(data_file)
    
    # 7th data analysis
#    dim7_data_analysis(data)

    # get pattern data index and test data index
    _, idx_train, idx_test = label_index_clf(data_file)
    # compute pattern
    pattern = compute_pattern(data, idx_train)
    # compute distance and similarity between pattern and test data
    dist_sep, dist_all, sim_sep, sim_all = compute_dist_sim(data, idx_test, pattern)

    # get test dataset
#    test = np.vstack((data[idx_test[0]], data[idx_test[1]], data[idx_test[2]]))
    test = []
    for i in range(3):
        test.append(data[idx_test[i]])
    
    # draw subplots
    draw_subplots(test, pattern)
