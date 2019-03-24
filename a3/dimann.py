#!/usr/bin/env python3

#   Mnemonic:   dimann.py
#   Abstract:   Artifical Neual Network analysis over diminsionally reduced data.
#
#               Data from the training dataset given on the command line is read
#               and separated into parameters (features) and label (classification).
#               The data is then reduced to c components (features) using each of
#               these methods:  
#                   - Truncataed SVD
#                   - Principle components analysis (PCA)
#                   - Fast Independent components analysis (ICA)
#                   - Gaussian Random Projections (GRP)
#
#               Each set of reduced data is split into a training and test set.
#
#               Following splitting, the MLP artifical neural network classifier
#               is created for the selected solver and given the training data
#               to fit.  Folling the training session, the solver is given the 
#               reserved data (reduced) for prediction. The results of each 
#               predication are then compared to the known classification values
#               extracted from the original data.   As an added comparative, the
#               MLP solver is run on the full set of input training data and is
#               validated on the reserved test data.
#
#               NOTE: it might seem logical to concat the train and test data 
#                   and reduce all of it before splitting.  However, some training
#                   data sets (crowd) have known noise while the related test
#                   data does not.  For this reason, using only the training data,
#                   and reserving some of it after reduction, is the only valid
#                   way to do this.  If the user knows that training data is 'pure'
#                   then comobine training and test data before, however the base
#                   evaluation is then invalid.
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
#           https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
#           https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA
#           https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection
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
from plotter import km_cplot


def usage() :
    printf( "usage: dimann.py [-c components] [-f] [-i iterations] [-r learn-rate] [-s stats] [-S solver]  train-path test-path\n" )
    printf( "\t use -f if the classification in the data is in col 0 (first), else it is assumed to be in the nth col\n" )
    printf( "\t solvers must be one of: all, adam, lbfgs or sgd. Defualt is all.\n" )
    printf( "\t learn rate must be one of: constant, adaptive, invscaling\n" )


# Nicely label the output.
#
def print_header( ) :
    printf( "# %5s  %6s  %6s  %6s  %5s %5s %5s\n", "Mthd", "Acc", "Prcsn", "Recall", "Elap", "nFeat", "Solvr" )

# Print the stats, just the stats.
#
def print_stats( rmethod, acc, elapsed, comps, precsn, recall, method ) :
    printf( "  %5s %6.2f%% %6.2f%% %6.2f%% %5ds %5d %5s\n", rmethod, acc*100, precsn*100, recall*100, elapsed, comps, method )

# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
# if you programme in go, then you recognise the beauty here :) 
flags = {                              # define possible flags and the default: map key, type, default value
    "-c": ("components", int_flag, 5),          # number of features (parameters) to reduce to (components isn't descriptive)
    "-f": ("classfirst", bool_flag, False),
    "-i": ("max-iterations", int_flag, 600),
    "-r": ("learn_rate", str_flag, "constant"), # must be one of constant, adaptive, invscaling
    "-s": ("stats", bool_flag, True),          # number of clusters to divide into
    "-S": ("solver", str_flag, "all"),         # solver (must be all, adam, lbfgs, or sgd)
    "-t": ("trials", int_flag, "1"),         # number of trials on fitting
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 1 :
    printf( "missing filename on command line (training data)\n" )
    sys.exit( 1 )

train_fn = pparms[0]              # file names; training validation (test)
#test_fn = pparms[1];

max_iters = opts["max-iterations"]
comps = opts["components"]          # number to reduce features/parameters down to
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
printf( "# data: %d instances  %d parameters reduction to %d features\n", data_n, data_p, comps )

reserve = int( data_n * .30 ) 

rdata = []              # reduced data for training
rtest_data = []         # reduced data reserved for testing
rmethod=[]              # method name for stats

bl_data = data[0:data_n-reserve,]         # save first 70% for baseline training
bl_vdata = data[reserve:,]                # the remaining 30% used for accuracy measure

rlabels = labels[:data_n-reserve,]        # split classification lables to match the split train/test data sizes
rvlabels = labels[reserve:,]

# ---- reduce the data using the various methods ---------------------------------------
printf( "# building methods...\n" );

svd = TruncatedSVD( n_components=comps, n_iter=13, random_state=1 )
all_reduced = svd.fit_transform( data )                 # reduce all data to c features
rdata.append( all_reduced[0:data_n-reserve,] )          # save first 70% for training
rtest_data.append( all_reduced[reserve:,] )             # the remaining 30% used for accuracy measure
rmethod.append( "svd" )

# ------------------------------------------------------------------------
pca = PCA( n_components=comps, whiten=True, random_state=1 )
pca.fit( train_data )                           # fit the model to training data, then transform the data
all_reduced = pca.transform( train_data )
rdata.append( all_reduced[0:data_n-reserve,] )
rtest_data.append( all_reduced[reserve:,] )
rmethod.append( "pca" )

# ------------------------------------------------------------------------
ica = FastICA( max_iter=500, n_components=comps, random_state=1 )
ica.fit( train_data )
all_reduced = ica.transform( train_data )
rdata.append( all_reduced[0:data_n-reserve,] )
rtest_data.append( all_reduced[reserve:,] )
rmethod.append( "ica" )

# ------------------------------------------------------------------------
grp = GaussianRandomProjection( n_components=comps, eps=0.1, random_state=1 )       # reduce data to n components
grp.fit( train_data )
all_reduced = grp.transform( train_data )
rdata.append( all_reduced[0:data_n-reserve,] )
rtest_data.append( all_reduced[reserve:,] )
rmethod.append( "grp" )


# ------ build the classifier, then run the various data through the neural net -----------------------------------
print_header( )

for si in range( len( solvers ) ) :
    solver = solvers[si]

    cfier = MLPClassifier( validation_fraction=.30, max_iter=max_iters, learning_rate=opts["learn_rate"], solver=solver  )
    ts_start = time()
    #cfier.fit( data, labels )                # baseline on the training data passed in (un-reduced)
    cfier.fit( bl_data, rlabels )             # baseline on the 70% unreduced data
    elapsed = time() - ts_start
    predicted = cfier.predict( bl_vdata )                                 # predict on the reserved training data
    accuracy = accuracy_score( rvlabels, predicted )                      # measure how well the prediction did using the validation labels
    prec = precision_score( rvlabels, predicted, average="micro" )        # measure of accuracy (higher means higher overall accuracy)
    recall = recall_score( rvlabels, predicted, average="micro" )         # measure of usefullness (higher means less positive misses)
    print_stats( "base", accuracy, elapsed, train_p-1, prec, recall, solver )


    for i in range( len( rdata ) ) :            # for each set of reduced feature data
        m = rmethod[i]
        d = rdata[i]
        td = rtest_data[i]                      # validation (data) reduced

        ts_start = time()
        cfier.fit( d, rlabels )                                               # fit using the reduced features
        elapsed = time() - ts_start
        predicted = cfier.predict( td )                                       # preduct labels on test data
        accuracy = accuracy_score( rvlabels, predicted )                      # measure how well the prediction did using the validation labels
        prec = precision_score( rvlabels, predicted, average="micro" )        # measure of accuracy (higher means higher overall accuracy)
        recall = recall_score( rvlabels, predicted, average="micro" )         # measure of usefullness (higher means less positive misses)

        print_stats( m, accuracy, elapsed, comps, prec, recall, solver )
        
