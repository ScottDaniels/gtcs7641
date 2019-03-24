
#   Mnemonic:   em.py
#   Abstract:   Run em (Expectation Maximisation)
#
#   Author:     E. Scott Danies
#   Date:       06 March 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#       code examples from the following URLs:
#           https://github.com/minmingzhao?tab=repositories
#
# ------------------------------------------------------------------

import sys
from time import time

import numpy as np
import scipy as sp
import pandas as pd

import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf,sprintf
from tools import parse_args,bool_flag,val_flag,str_flag,int_flag
from tools import print_pd
from plotter import c_cplot,gt_cplot


def usage() :
    printf( "usage: kcluster.py [-f] [-i iterations] [-k k-value] train-path test-path\n" )

# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
# if you programme in go, then you recognise the beauty here :) 
flags = {                              # define possible flags and the default: map key, type, default value
    "-d": ("output-dir", str_flag, "/tmp"),
    "-f": ("classfirst", bool_flag, False),
    "-i": ("iterations", int_flag, 10),
    "-k": ("k-components", int_flag, 2),          # number of clusters to divide into
    "-s": ("plot-samp-rate", int_flag, 10)          # to keep plot output sizes reasonable, sample at x% for plots
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 2 :
    printf( "missing filenames on command line (training testing)\n" )
    sys.exit( 1 )

train_fn = pparms[0]              # file names; training validation (test)
test_fn = pparms[1];

components = opts["k-components"]
out_dir = opts["output-dir"]

np.random.seed( 17 )
# -----------------------------------------------------------------------------------------
train_data = pd.read_csv( train_fn, sep= ',' )  # suck in datasets
test_data = pd.read_csv( test_fn, sep= ',' )
train_n, train_p = train_data.shape          # number of training instances and parameters
test_n, test_p = test_data.shape

if opts["classfirst"] :                     # class target value is in col 0
    data = train_data.iloc[:,1:train_p]
    labels = train_data.values[:,0]               # get just the first column which has 'truth'
else :
    data = train_data.iloc[:,0:train_p-1]
    labels = train_data.values[:,-1]               # get just the last column which has 'truth'

data = data.values
data_n, data_p = data.shape          # data instances and parameters
printf( "data: %d instances  %d parameters\n", data_n, data_p )


#--------------------------------------------------------------------------------------------
printf( "#%-5s %-5s %-5s %-5s %-5s %-5s %-5s\n", "ACC", "HOMO", "COMPL", "VM", "ARAND", "MI", "CH-idx" )

for i in range( opts["iterations"] ) :
    em = GaussianMixture( n_components=components, n_init=13, covariance_type="full" ).fit( data )
    guess = em.predict( data )

    acc = metrics.accuracy_score( labels, guess )
    homo = metrics.homogeneity_score( labels, guess )         # compare the true lables to those em predicted
    comp = metrics.completeness_score( labels, guess )
    vm = metrics.v_measure_score( labels, guess ) 
    arand = metrics.adjusted_rand_score( labels, guess )
    mi = metrics.adjusted_mutual_info_score( labels,  guess, average_method="arithmetic" )
    ch = metrics.calinski_harabaz_score( data, guess );

    printf( " %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n", acc, homo, comp, vm, arand, mi, ch)

    if i == 0 :             # just plot the first
        tokens = train_fn.split( "/" );             # build file name as emax_<data-type>_<clusters>.eps
        tokens = tokens[-1].split( "_" )
        title = sprintf( "Exp Max %s k=%d", tokens[0], components ) 
        gfname = sprintf( "%s/emax_%s_%d.eps", out_dir, tokens[0], components  )

        # pretend guess is ground truth and plot predicted cluster
        gt_cplot( data, guess, components, gfname, title, sample_pct=opts["plot-samp-rate"], cpattern=2 )    

sys.exit( 0 )


