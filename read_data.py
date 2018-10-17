#!/usr/bin/python

import re
import numpy as np
import pandas as pd

data_file = "./touch_events.csv"

# Load data from appointed file
def load_data(filename):
    # load raw *.csv data
    src_df = pd.read_csv(filename, sep=",")
    
    # get column names of src_df
    cols = src_df.columns

    # get 'event' column data from raw data
    i = 0
    val = src_df['event'].values
    new_val = []
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
    
    return new_val

# Find index of each kind of data
def label_index_clf(filename):
    # load raw *.csv data
    src_df = pd.read_csv(filename, sep=",")
    cols = src_df.columns

    # get 'class' column data from raw data
    label = src_df['class'].values
    # initial label index data : 0(b), 1(k), 2(r)
    idx = [[],[],[]]
    idx_count = np.zeros(3)
    for i in range(len(label)):
        ll = label[i]
        if ll == 'b':
            idx[0].append(i)
            idx_count[0] += 1
        elif ll == 'k':
            idx[1].append(i)
            idx_count[1] += 1
        elif ll == 'r':
            idx[2].append(i)
            idx_count[2] += 1
        else:
            print ("Unexpected label {}".format(label[i]))
            break

    # seperate idx into training set and testing set
    idx_train = idx.copy()
    idx_train[0] = idx[0][:20]
    idx_train[1] = idx[1][:20]
    idx_train[2] = idx[2][:20]
    idx_test = idx.copy()
    idx_test[0] = idx[0][20:]
    idx_test[1] = idx[1][20:]
    idx_test[2] = idx[2][20:]

    return idx, idx_train, idx_test

# Select each kind of label data and compute pattern of each kind of data
def compute_pattern(data, idx):
    # get data of each kind 
    # compute mean data of dim2, that is average 8 sensors' data
    data_b = np.mean(data[idx[0]], axis=0)
    data_k = np.mean(data[idx[1]], axis=0)
    data_r = np.mean(data[idx[2]], axis=0)
    
    # compute pattern of each kind
    pattern = np.zeros((3, data_b.shape[0]))
    pattern[0] = np.mean(data_b, axis=1)
    pattern[1] = np.mean(data_k, axis=1)
    pattern[2] = np.mean(data_r, axis=1)

    return pattern

# Compute distance and similarity between pattern and test data
def compute_dist_sim(data, idx, pattern):
    ## get three kind of data seperately : 0(b), 1(k), 2(r)
    #data_b = data[idx[0]]
    #data_k = data[idx[1]]
    #data_r = data[idx[2]]

    # compute distance of three kind data
    dist = []
    sim  = []
    dist_bkr = []
    sim_bkr  = []
    for i in range(len(idx)):
        test_data = data[idx[i]]
        test_patn = pattern[i,:]
        #d0, d1, d2 = test_data.shape
        sample_num, time, sensor_num = test_data.shape
        err = np.zeros((sample_num, sensor_num, time))
        dist_sep = np.zeros((sample_num, sensor_num))
        dist_all = np.zeros((sample_num))
        sim_sep  = np.zeros((sample_num, sensor_num))
        sim_all  = np.zeros((sample_num))
        for j in range(sample_num):
            for l in range(sensor_num):
                for k in range(time):
                    err[j,l,k] = test_data[j,k,l] - test_patn[k]
                    dist_sep[j,l] += np.square(err[j,l,k])
                dist_sep[j,l] = np.sqrt(dist_sep[j,l])
                sim_sep[j,l]  = 1 / dist_sep[j,l]
            dist_all[j] = np.mean(dist_sep[j])
            sim_all[j]  = 1 / dist_all[j]
        dist.append(dist_sep)
        dist_bkr.append(dist_all)
        sim.append(sim_sep)
        sim_bkr.append(sim_all)

    # get each kind of distance between pattern and test data
    dist_b = dist[0]
    dist_k = dist[1]
    dist_r = dist[2]

    # compute similarity

    return dist, dist_bkr, sim, sim_bkr

if __name__ == "__main__":
    # load sensor data
    data = load_data(data_file)
    # get pattern data index, and test data index
    _, idx_train, idx_test = label_index_clf(data_file)
    # compute pattern 
    pattern = compute_pattern(data, idx_train)
    # compute distance and similarity between pattern and test data
    dist_sep, dist_all, sim_sep, sim_all = compute_dist_sim(data, idx_test, pattern)
