#!/usr/bin/env python3

#   Mnemonic:   dimred.py
#   Abstract:   Dimension reduction over a dataset
#
#   Author:     E. Scott Danies
#   Date:       05 March 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#       code examples from the following URLs:
#           https://github.com/minmingzhao?tab=repositories
#           https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
#           https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA
            #https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html#sklearn.random_projection.GaussianRandomProjection
#
# ------------------------------------------------------------------

import sys
from time import time

import matplotlib
matplotlib.use('Agg')      # because this makes all kinds of sense; must be called before the pyplot import! WTF python
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

# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf,sprintf
from tools import parse_args,bool_flag,val_flag,str_flag,int_flag
from tools import print_pd
from tools import run_kmeans
from plotter import km_cplot


def usage() :
    printf( "usage: dimred.py [-c components] [-f] [-i iterations] [-k k-value] train-path test-path\n" )

def head( arrays, n=10 ) :
    for i in range( n ) :
        print( arrays[i] )

#
#   Print the header for the columns that run_kmeans spits out
def km_header( nfeatures ) :
    printf( "\n" )
    printf( "# features kept: %d\n", nfeatures )
    printf( "#%6s %6s %6s %6s %6s %6s %6s %10s %s\n", "Mthod", "ACC", "HOMO", "COMPL", "VM", "ARAND", "MI", "CH-scr", "N-clus" )


# given a method and data path name, generate a file prefix
#
def mk_prefix( method, dname, clusters, comps ) :
    tokens = dname.split( "/" )
    tokens = train_fn[-1].split( "_" )
    fprefix = sprintf( "%s_%s", method, file_id )

    return fprefix

# 
#   run the data through the em process and gen stats
def run_em( data, labels, iterations, clusters, method ) :
    for c in range( len( clusters ) ) :
        for i in range( iterations ) :
            em = GaussianMixture( n_components=clusters[c], n_init=13, covariance_type="full" ).fit( data )
            guess = em.predict( data )

            acc = metrics.accuracy_score( labels, guess )
            homo = metrics.homogeneity_score( labels, guess )         # compare the true lables to those em predicted
            comp = metrics.completeness_score( labels, guess )
            vm = metrics.v_measure_score( labels, guess ) 
            arand = metrics.adjusted_rand_score( labels, guess )
            mi = metrics.adjusted_mutual_info_score( labels,  guess, average_method="arithmetic" )
            if c > 1 :
                ch = metrics.calinski_harabaz_score( data, guess );
            else :
                ch = 0

            printf( " %6s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6d em\n", method, acc, homo, comp, vm, arand, mi, ch, clusters[c] )

        if iterations > 1 :
            printf( "\n" )

# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
# if you programme in go, then you recognise the beauty here :) 
flags = {                              # define possible flags and the default: map key, type, default value
    "-c": ("components", int_flag, 5),          # number of features (parameters) to reduce to (components isn't descriptive)
    "-f": ("classfirst", bool_flag, False),
    "-i": ("iterations", int_flag, 10),
    "-k": ("k-clusters", int_flag, 0),          # override cvalues with a single k-cluster setting
    "-s": ("stats", bool_flag, True),          # number of clusters to divide into
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 2 :
    printf( "missing filenames on command line (training testing)\n" )
    sys.exit( 1 )

train_fn = pparms[0]              # file names; training validation (test)
test_fn = pparms[1];

clusters = opts["k-clusters"]
comps = opts["components"]          # number to reduce features/parameters down to
show_stats = opts["stats"]

np.random.seed( 17 )
# -----------------------------------------------------------------------------------------
train_data = pd.read_csv( train_fn, sep= ',' )  # suck in datasets
test_data = pd.read_csv( test_fn, sep= ',' )
train_n, train_p = train_data.shape          # number of training instances and parameters
test_n, test_p = test_data.shape

if opts["classfirst"] :                     # class target value is in col 0
    data = train_data.iloc[:,1:train_p]
    labels = train_data.values[:,0]               # get just the first column which has 'truth'
    vdata = test_data.iloc[:,1:train_p]
    vlabels = test_data.values[:,0]               # get just the first column which has 'truth'
else :
    data = train_data.iloc[:,0:train_p-1]
    labels = train_data.values[:,-1]               # get just the last column which has 'truth'
    vdata = test_data.iloc[:,0:train_p-1]
    vlabels = test_data.values[:,-1]               # get just the first column which has 'truth'

data = data.values
data_n, data_p = data.shape          # data instances and parameters
printf( "# data: %d instances  %d parameters\n", data_n, data_p )


if opts["k-clusters"] > 0 :
    cvalues = [ opts["k-clusters" ] ]         # just the user supplied k
else :
    cvalues = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]            # run kmans for these cluster values

tokens = train_fn.split( "/" )
tokens = tokens[-1].split( "_" )
file_id = tokens[0]


# --- apply various dimension reduction methods to the data -------------------------------

# --------------------------------------------------------------------------------------
svd = TruncatedSVD( n_components=comps, n_iter=13, random_state=1 )
svd_xf_train = svd.fit_transform( train_data )
fprefix = mk_prefix( "svd", train_fn, clusters, comps )
km_header( comps )
run_kmeans( svd_xf_train, labels, comps, cvalues=cvalues, iterations=opts["iterations"], 
    tprefix="SVD:", fprefix=fprefix, gen_stats=show_stats, method="svd" )
run_em( data, labels, opts["iterations"], cvalues, "svd" )

# --------------------------------------------------------------------------------------
pca = PCA( n_components=comps, whiten=True )
pca.fit( train_data )                           # fit the model to training data

pca_xf_train = pca.transform( train_data )      # transform training and test data
fprefix = mk_prefix( "pca", train_fn, clusters, comps )
km_header( comps )
run_kmeans( pca_xf_train, labels, comps, cvalues=cvalues, iterations=opts["iterations"], 
    tprefix="PCA:", fprefix=fprefix, gen_stats=show_stats, method="pca" )
run_em( data, labels, opts["iterations"], cvalues, "pca" )
#print( pca.components_ )

# --------------------------------------------------------------------------------------
#ica = FastICA( n_components=comps, whiten=False )
#ica.fit( pca_xf_train )     # train on the pca generated (whitened) data
ica = FastICA( n_components=comps )
ica.fit( train_data )

ica_xf_train = ica.transform( train_data )      # reduce to n components
#ica_xf_train = ica.transform( pca_xf_train )      # reduce to n components
fprefix = mk_prefix( "ica", train_fn, clusters, comps )
km_header( comps )
run_kmeans( ica_xf_train, labels, comps, cvalues=cvalues, iterations=opts["iterations"], 
    tprefix="ICA:", fprefix=fprefix, gen_stats=show_stats, method="ica" )
run_em( data, labels, opts["iterations"], cvalues, "ica" )


# --------------------------------------------------------------------------------------
grp = GaussianRandomProjection( n_components=comps, eps=0.1 )       # reduce data to n components
grp.fit( train_data )
grp_xf_train = grp.transform( train_data )
fprefix = mk_prefix( "ica", train_fn, clusters, comps )

km_header( comps )
run_kmeans( grp_xf_train, labels, comps, cvalues=cvalues, iterations=opts["iterations"], 
    tprefix="GRP:", fprefix=fprefix, gen_stats=show_stats, method="grp" )
run_em( data, labels, opts["iterations"], cvalues, "grp" )



#  --- these were just broken for variouis reasons -------------------------------------
# fit threw an error about a missing parameter y even if y=None given.
#ld = LinearDiscriminantAnalysis( solver="svd", n_components=comps )
#ld_xf_train = ld.fit_transform( train_data, y=None )

# used huge amounts of memory -- avoid
#se = SpectralEmbedding( n_components=comps )
#se_xf_train = se.fit_transform( train_data )
