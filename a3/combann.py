#!/usr/bin/env python3

#   Mnemonic:   combann.py
#   Abstract:   Experiments which combine cluster assignments for the data
#               with the data, and generate the peformance of the MLP ANN
#               algorithm on that data.
#
#               Data from the data set given on the command line is read and 
#               separated into parameters (features) and label (classification).
#               The data is then clustered using both k-means and E-max. The clusters
#               generated by these algorithms is added to the original data as new 
#               features generating the "cluster enriched data." The cluster enriched
#               data is then used as input to the ANN.
#       
#               The number of clusters that the data is to be divided into is given on 
#               the command line.  This value should be predetermined by running a 
#               clustering algorithm and using something like the CH-index to determine
#               the optimal number of clusters.
#
#               Because the data will be agumented, only the training set should be  used.
#               Approximately 30% of the data, after combination, is removed and used as the
#               test data for validation; this data is not used to train the ANN.
#
#   Author:     E. Scott Danies
#   Date:       15 March 2019
#
#   Acknowledgements: 
#       This code is largly orignal, however the initial ideas were taken from 
#       the code in the following repository:
#           https://github.com/minmingzhao?tab=repositories
#
#       These sklearn pages were also crutial in creating this code
#           https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
#
# ------------------------------------------------------------------

import sys
from time import time

import matplotlib
matplotlib.use('Agg')      # because this makes all kinds of sense (not); must be called before the pyplot import! WTF python
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
#from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# dim reduction specific
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

# classifier used in assignment 1
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf,sprintf
from tools import parse_args,bool_flag,val_flag,str_flag,int_flag
from tools import print_pd


def usage() :
    printf( "usage: combann.py [-f] [-i iterations] [-r learn-rate] [-s stats] [-S solver]  train-path\n" )
    printf( "\t use -f if the classification in the data is in col 0 (first), else it is assumed to be in the nth col\n" )
    printf( "\t solvers must be one of: all, adam, lbfgs or sgd. Defualt is all.\n" )
    printf( "\t learn rate must be one of: constant, adaptive, invscaling\n" )


# Nicely label the output
#
def print_header( ) :
    printf( "# %5s  %6s  %6s  %6s  %5s %5s %5s\n", "Mthd", "Acc", "Prcsn", "Recall", "Elap", "Kclust", "Solvr" )

# Print the stats, just the stats.
#
def print_stats( rmethod, acc, elapsed, kclusters, precsn, recall, method ) :
    printf( "  %5s %6.2f%% %6.2f%% %6.2f%% %5ds %5d %5s\n", rmethod, acc*100, precsn*100, recall*100, elapsed, kclusters, method )

# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
# if you programme in go, then you recognise the beauty here :) 
flags = {                              # define possible flags and the default: map key, type, default value
    "-f": ("classfirst", bool_flag, False),
    "-i": ("max-iterations", int_flag, 600),
    "-k": ("k-clusters", int_flag, 5),          # number of clusters to generate and add as features
    "-r": ("learn_rate", str_flag, "constant"), # must be one of constant, adaptive, invscaling
    "-s": ("stats", bool_flag, True),          # number of clusters to divide into
    "-S": ("solver", str_flag, "all"),         # solver (must be all, adam, lbfgs, or sgd)
    "-t": ("trials", int_flag, "1"),         # number of trials on fitting
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 1 :
    printf( "missing filenames on command line (training testing)\n" )
    sys.exit( 1 )

train_fn = pparms[0]              # file names; training validation (test)

max_iters = opts["max-iterations"]
kclusters = opts["k-clusters"]          # number to reduce features/parameters down to
show_stats = opts["stats"]

if opts["solver"] == "all" :                    # configure list of solvers based on command line or defaul (all)
    solvers = [ "adam", "lbfgs", "sgd" ]        # list of solvers that we'll run later
else :
    solvers = [ opts["solver"] ]

np.random.seed( 17 )
# -----------------------------------------------------------------------------------------
train_data = pd.read_csv( train_fn, sep= ',' )  # suck in datasets
train_n, train_p = train_data.shape          # number of training instances and parameters

if opts["classfirst"] :                     # class target value is in col 0
    data = train_data.iloc[:,1:train_p]
    labels = train_data.values[:,0]               # get just the first column which has 'truth'
else :
    data = train_data.iloc[:,0:train_p-1]
    labels = train_data.values[:,-1]               # get just the last column which has 'truth'

data = data.values
data_n, data_p = data.shape          # data instances and parameters
printf( "# data: %d instances  %d parameters using k=%d for clustering\n", data_n, data_p, kclusters )

reserve = int( data_n * .30 )       # number of records pulled out and used for validation

adata = []              # augmented data for training
atest_data= []          # augmented data reserved for validation (testing)
rmethod=[]              # clustering method name for stats

# ---------- run kmeans to generate the cluster assignments ----------------------
kmeans = KMeans( n_clusters=kclusters, init="k-means++", n_init=1,  max_iter=100  )
kmeans.fit( data ) 
kmcd = np.insert( data, 0, kmeans.labels_, axis=1 )    # add cluster labels as a feature

adata.append( kmcd[0:data_n-reserve,] )          # save first n% for training
atest_data.append( kmcd[reserve:,] )            # the remaining 100-n% used for accuracy measure
rmethod.append( "kmeans" )

# ---------- generate cluster groups with em -------------------------------------
if True :
    em = GaussianMixture( n_components=kclusters, n_init=13 )
    em.fit( data )
    emp = em.predict( data )
    emcd = np.insert( data, 0, emp, axis=1 ) # add clustering information
    
    adata.append( emcd[0:data_n-reserve,] )         # save first n% for training
    atest_data.append( emcd[reserve:,] )            # the remaining 100-n% used for accuracy measure
    rmethod.append( "emax" )


# ------ build the classifier, then run the various data through the neural net -----------------------------------
print_header( )

for si in range( len( solvers ) ) :
    solver = solvers[si]

    cfier = MLPClassifier( validation_fraction=.30, max_iter=max_iters, learning_rate=opts["learn_rate"], solver=solver  )
    #ts_start = time()
    #cfier.fit( data, labels )             # baseline on the training data passed in (un-reduced)
    #elapsed = time() - ts_start
    #predicted = cfier.predict( vdata )                           # preduct labels on test data
    #accuracy = accuracy_score( vlabels, predicted )                # measure how well the prediction did using the validation labels
    #prec = precision_score( vlabels, predicted, average="micro" )        # measure of accuracy (higher means higher overall accuracy)
    #recall = recall_score( vlabels, predicted, average="micro" )         # measure of usefullness (higher means less positive misses)
    #print_stats( "base", accuracy, elapsed, train_p, prec, recall, solver )

    rlabels = labels[:data_n-reserve,]          # split classification lables to match the split train/test data sizes
    rvlabels = labels[reserve:,]

    for i in range( len( adata ) ) :            # for each set of reduced feature data
        m = rmethod[i]
        d = adata[i]
        td = atest_data[i]                      # validation (data) reduced

        ts_start = time()
        cfier.fit( d, rlabels )                                          # fit using the reduced features
        elapsed = time() - ts_start
        predicted = cfier.predict( td )                                  # preduct labels on test data
        accuracy = accuracy_score( rvlabels, predicted )                 # measure how well the prediction did using the validation labels
        prec = precision_score( rvlabels, predicted, average="micro" )        # measure of accuracy (higher means higher overall accuracy)
        recall = recall_score( rvlabels, predicted, average="micro" )         # measure of usefullness (higher means less positive misses)

        print_stats( m, accuracy, elapsed, kclusters, prec, recall, solver )
        
