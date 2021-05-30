#!/usr/bin/env python
# coding: utf-8

# Import header files
import math
import argparse
import torch
from sklearn import preprocessing
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import sys
import numpy as np
import pylab
from matplotlib import pyplot as plt
import time
import sys
from models.sru import SRU, trainSRU
from models.esru_1LF import eSRU_1LF, train_eSRU_1LF
from models.esru_2LF import eSRU_2LF, train_eSRU_2LF
import pandas as pd
from util import *
from utils.utilFuncs import env_config, loadTrainingData, loadTrueNetwork, getCausalNodes, count_parameters, \
    getGeneTrainingData

# Read input command line arguments
parser = argparse.ArgumentParser()

# https://github.com/xiangyu-sun-789/SRU_for_GCI#input-arguments
parser.add_argument('--device', type=str, default='cuda:3',
                    help='device, default: cuda:3')
# parser.add_argument('--dataset', type=str, default='VAR',
#                      help='dataset type, default: VAR')
# parser.add_argument('--dsid', type=int, default=1,
#                      help='dataset id, default: 1')
# parser.add_argument('--T', type=int, default=10,
#                      help='training size, default: 10')
parser.add_argument('--F', type=int, default=10,
                    help='chaos, default: 10')
# parser.add_argument('--n', type=int, default=10,
#                      help='num of timeseries, default: 10')
parser.add_argument('--model', type=str, default='sru',
                    help='[sru, gru, lstm]: select your model')
parser.add_argument('--nepochs', type=int, default=500,
                    help='sets max_iter, default: 500')
parser.add_argument('--mu1', type=float, default=0.5,
                    help='Bias for ridge regularization of all unregularized weights in the model, default: 1')
parser.add_argument('--mu2', type=float, default=0.5,
                    help='Bias for block sparse regularization of input layer weights, default: 1')
parser.add_argument('--mu3', type=float, default=0.5,
                    help='Bias for group sparse regularization of output feature layer weights in Economy SRU, default: 1')
parser.add_argument('--lr', type=float, default=0.005,
                    help='sets learning rate, default: 0.005')
parser.add_argument('--joblog', type=str, default="",
                    help='name of job logfile, default=""')

args = parser.parse_args()
deviceName = args.device
model_name = args.model
max_iter = args.nepochs
mu1 = args.mu1
mu2 = args.mu2
mu3 = args.mu3
# dataset    = args.dataset
# dataset_id = args.dsid
# T          = args.T
F = args.F
# n          = args.n
lr = args.lr
jobLogFilename = args.joblog

###############################
# Global simulation settings
###############################
verbose = 0  # Verbosity level

#################################
# Pytorch environment
#################################
device, seed = env_config(True, deviceName)  # true --> use GPU
print("Computational Resource: %s" % (device))

######################################
# Create input data in batch format 
######################################

# if(dataset == 'gene'):
#     Xtrain, Gref = getGeneTrainingData(dataset_id, device)
#     n1 = Xtrain.shape[0]
#     if(n != n1):
#         print("Error::Dimension mismatch for input training data..")
#     numTotalSamples = Xtrain.shape[1]
#     Xtrain = Xtrain.float().to(device)
#     # Make input signal zero mean and appropriately scaled
#     Xtrain = Xtrain - Xtrain.mean()
#     inputSignalMultiplier = 50
#     Xtrain = inputSignalMultiplier * Xtrain
#
# elif(dataset == 'var'):
#     fileName = "data/var/S_%s_T_%s_dataset_%s.npz" % (F, T, dataset_id)
#     ld = np.load(fileName)
#     X_np = ld['X_np']
#     Gref = ld['Gref']
#     numTotalSamples = T
#     Xtrain = torch.from_numpy(X_np)
#     Xtrain = Xtrain.float().to(device)
#     inputSignalMultiplier = 1
#     Xtrain = inputSignalMultiplier * Xtrain

# elif(dataset == 'lorenz'):
# fileName = "data/lorenz96/F_%s_T_%s_dataset_%s.npz" % (F, T, dataset_id)
fileName = "/Users/shawnxys/Development/Data/preprocessed_causal_sports_data_by_games/17071/features_shots_rewards.csv"
# ld = np.load(fileName)
# X_np = ld['X_np']
# Gref = ld['Gref']

features_shots_rewards_df = pd.read_csv(fileName)
# rename column name
features_shots_rewards_df = features_shots_rewards_df.rename(columns={'reward': 'goal'})

X = features_shots_rewards_df.to_numpy()

# data standardization
scaler = preprocessing.StandardScaler().fit(X)
normalized_X = scaler.transform(X)

print('feature std after standardization: ', normalized_X.std(axis=0))
assert (normalized_X.std(axis=0).round(
    decimals=3) == 1).all()  # make sure all the variances are (very close to) 1

X_np = normalized_X.transpose()

n = X_np.shape[0]  # No. of timeseries/Nodes
T = X_np.shape[1]  # Length of input timeseries

# input data shape should be (number of variables, number of time steps)
assert n == 12 and T == 4021

numTotalSamples = T
Xtrain = torch.from_numpy(X_np)
Xtrain = Xtrain.float().to(device)
inputSignalMultiplier = 1
Xtrain = inputSignalMultiplier * Xtrain

# elif(dataset == 'netsim'):
#     fileName = "data/netsim/sim3_subject_%s.npz" % (dataset_id)
#     ld = np.load(fileName)
#     X_np = ld['X_np']
#     Gref = ld['Gref']
#     numTotalSamples = T
#     Xtrain = torch.from_numpy(X_np)
#     Xtrain = Xtrain.float().to(device)
#     inputSignalMultiplier = 1
#     Xtrain = inputSignalMultiplier * Xtrain
#
# else:
#     print("Dataset is not supported")


if (verbose >= 1):
    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("x0(t)")
    plt.plot(range(numTotalSamples), Xtrain.cpu().numpy()[0][:])
    plt.show(block=False)
    plt.pause(0.1)

######################################
# SRU Cell parameters
######################################


#######################################
# Model training parameters
######################################
if (model_name == 'sru'):

    lr_gamma = 0.99
    lr_update_gap = 4
    staggerTrainWin = 1
    stoppingThresh = 1e-5;
    trainVerboseLvl = 2
    lr = lr
    lambda1 = mu1
    lambda2 = mu2
    n_inp_channels = n
    n_out_channels = 1

    # if(dataset == 'gene'):
    #     A = [0.0, 0.01, 0.1, 0.5, 0.99]; #0.75
    #     dim_iid_stats = 10 #math.ceil(n) #1.5n
    #     dim_rec_stats = 10 #math.ceil(n) #1.5n
    #     dim_final_stats = 10 #d * len(A) #math.ceil(n/2)
    #     dim_rec_stats_feedback = 10 #d * len(A)
    #     batchSize = 21
    #     blk_size = batchSize
    #     numBatches = int(numTotalSamples/batchSize)
    #
    #
    # elif(dataset == 'var'):
    #     A = [0.0, 0.01, 0.1, 0.99];
    #     dim_iid_stats = 10 #math.ceil(n) #1.5n
    #     dim_rec_stats = 10 #math.ceil(n) #1.5n
    #     dim_final_stats = 10 #d * len(A) #math.ceil(n/2) #n
    #     dim_rec_stats_feedback = 10 #d * len(A) #math.ceil(n/2) #n
    #     batchSize = 250
    #     blk_size = int(batchSize/2)
    #     numBatches = int(numTotalSamples/batchSize)
    #
    #
    # elif(dataset == 'lorenz'):
    A = [0.0, 0.01, 0.1, 0.99];
    dim_iid_stats = 10
    dim_rec_stats = 10
    dim_final_stats = 10
    dim_rec_stats_feedback = 10
    batchSize = 250
    blk_size = int(batchSize / 2)
    numBatches = int(numTotalSamples / batchSize)

    # elif(dataset == 'netsim'):
    #     A = [0.0, 0.01, 0.05, 0.1, 0.99];
    #     dim_iid_stats = 10
    #     dim_rec_stats = 10
    #     dim_final_stats = 10
    #     dim_rec_stats_feedback = 10
    #     batchSize = 10 #100
    #     blk_size = int(batchSize/2)
    #     numBatches = int(numTotalSamples/batchSize)
    #
    # else:
    #     print("Unsupported dataset encountered")

elif (model_name == 'eSRU_1LF' or model_name == 'eSRU_2LF'):

    lr_gamma = 0.99
    lr_update_gap = 4
    staggerTrainWin = 1
    stoppingThresh = 1e-5;
    trainVerboseLvl = 2
    lr = lr
    lambda1 = mu1
    lambda2 = mu2
    lambda3 = mu3
    n_inp_channels = n
    n_out_channels = 1

    # if(dataset == 'gene'):
    #     A = [0.05, 0.1, 0.2, 0.99];
    #     dim_iid_stats = 10
    #     dim_rec_stats = 10
    #     dim_final_stats = 10
    #     dim_rec_stats_feedback = 10
    #     batchSize = 21
    #     blk_size = int(batchSize)
    #     numBatches = int(numTotalSamples/batchSize)
    #
    #
    # elif(dataset == 'var'):
    #     A = [0.0, 0.01, 0.1, 0.99];
    #     dim_iid_stats = 10 #math.ceil(n) #1.5n
    #     dim_rec_stats = 10 #math.ceil(n) #1.5n
    #     dim_final_stats = 10 #d * len(A) #math.ceil(n/2) #n
    #     dim_rec_stats_feedback = 10 #d * len(A) #math.ceil(n/2) #n
    #     batchSize = 250
    #     blk_size = int(batchSize/2)
    #     numBatches = int(numTotalSamples/batchSize)
    #
    #
    # elif(dataset == 'lorenz'):
    # lr = 0.01
    A = [0.0, 0.01, 0.1, 0.99];
    dim_iid_stats = 10
    dim_rec_stats = 10
    dim_final_stats = 10  # d*len(A)
    dim_rec_stats_feedback = 10  # d*len(A)
    batchSize = 250
    blk_size = int(batchSize / 2)
    numBatches = int(numTotalSamples / batchSize)

    # elif(dataset == 'netsim'):
    #     A = [0.0, 0.01, 0.1, 0.99];
    #     dim_iid_stats = 10
    #     dim_rec_stats = 10
    #     dim_final_stats = 10 #d*len(A)
    #     dim_rec_stats_feedback = 10 #d*len(A)
    #     batchSize = 10 #10 #100
    #     blk_size = int(batchSize/2)
    #     numBatches = int(numTotalSamples/batchSize)
    #
    # else:
    #     print("Unsupported dataset encountered")

else:
    print("Unsupported model encountered")

############################################
# Evaluate ROC plots (regress mu2)
############################################
if 1:
    Gest = torch.zeros(n, n, requires_grad=False)

    if (model_name == 'sru'):
        for predictedNode in range(n):
            start = time.time()
            print("node = %d" % (predictedNode))
            model = SRU(n_inp_channels, n_out_channels, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                        dim_final_stats, A, device)
            model.to(device)  # shift to CPU/GPU memory
            print(count_parameters(model))
            model, lossVec = trainSRU(model, Xtrain, device, numBatches, batchSize, blk_size, predictedNode, max_iter,
                                      lambda1, lambda2, lr, lr_gamma, lr_update_gap, staggerTrainWin, stoppingThresh,
                                      trainVerboseLvl)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)
            print("Elapsed time (1) = % s seconds" % (time.time() - start))

    elif (model_name == 'eSRU_1LF'):
        for predictedNode in range(n):
            start = time.time()
            print("node = %d" % (predictedNode))
            model = eSRU_1LF(n_inp_channels, n_out_channels, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                             dim_final_stats, A, device)
            model.to(device)  # shift to CPU/GPU memory
            print(count_parameters(model))
            model, lossVec = train_eSRU_1LF(model, Xtrain, device, numBatches, batchSize, blk_size, predictedNode,
                                            max_iter,
                                            lambda1, lambda2, lambda3, lr, lr_gamma, lr_update_gap, staggerTrainWin,
                                            stoppingThresh, trainVerboseLvl)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)
            print("Elapsed time (1) = % s seconds" % (time.time() - start))

    elif (model_name == 'eSRU_2LF'):
        for predictedNode in range(n):
            start = time.time()
            print("node = %d" % (predictedNode))
            model = eSRU_2LF(n_inp_channels, n_out_channels, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                             dim_final_stats, A, device)
            model.to(device)  # shift to CPU/GPU memory
            print(count_parameters(model))
            model, lossVec = train_eSRU_2LF(model, Xtrain, device, numBatches, batchSize, blk_size, predictedNode,
                                            max_iter,
                                            lambda1, lambda2, lambda3, lr, lr_gamma, lr_update_gap, staggerTrainWin,
                                            stoppingThresh, trainVerboseLvl)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)
            print("Elapsed time (1) = % s seconds" % (time.time() - start))

    else:
        print("Unsupported model encountered")

    # print(Gref)
    print(Gest)

    # if(jobLogFilename != ""):
    #     if(model_name == 'eSRU_1LF' or model_name == 'eSRU_2LF'):
    #         np.savez(jobLogFilename,
    #                  Gref=None,
    #                  Gest=Gest.detach().cpu().numpy(),
    #                  model=model_name,
    #                  dataset=None,
    #                  dsid=None,
    #                  T=T,
    #                  F=F,
    #                  nepochs=max_iter,
    #                  mu1=mu1,
    #                  mu2=mu2,
    #                  mu3=mu3,
    #                  lr=lr,
    #                  batchSize=batchSize,
    #                  blk_size=blk_size,
    #                  numBatches=numBatches,
    #                  dim_iid_stats=dim_iid_stats,
    #                  dim_rec_stats=dim_rec_stats,
    #                  dim_final_stats=dim_final_stats,
    #                  dim_rec_stats_feedback=dim_rec_stats_feedback)
    #
    #     else:
    #         np.savez(jobLogFilename, Gref=None, Gest=Gest.detach().cpu().numpy(), model=model_name, dataset=None, dsid=None, T=T, F=F, nepochs=max_iter, mu1=mu1, mu2=mu2, lr=lr)

# sleep for one seconds followed by printing
# the exit key for tmux consumption
time.sleep(1)
print("#RUN_COMPLETE #RUN_COMPLETE #RUN_COMPLETE #RUN_COMPLETE")

# variable_names = ['x_' + str(i) for i in range(args.n)]
variable_names = [s for s in features_shots_rewards_df.columns]
print(variable_names)

csv_file_name = jobLogFilename + '.DAG.csv'

save_adjacency_matrix_in_csv(csv_file_name, Gest, variable_names)

draw_DAGs_using_LINGAM(csv_file_name, Gest, variable_names)
