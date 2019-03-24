
#   Mnemonic:   kclust.py
#   Abstract:   Run k-clustering on the provided data.
#
#   Author:     E. Scott Danies
#   Date:       05 March 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#       code examples from the following URLs:
#           https://github.com/minmingzhao?tab=repositories
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
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
#from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf,sprintf
from tools import parse_args,bool_flag,val_flag,str_flag,int_flag
from tools import print_pd
from plotter import km_cplot,gt_cplot


def usage() :
    printf( "usage: kcluster.py [-f] [-i iterations] [-k k-value] train-path test-path\n" )

# this plot code comes pretty much from the example code at sklearn.org:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#
def xptest( data, clusters, fname, title ) :
    reduced_data = PCA( n_components=2 ).fit_transform( data )
    #kmeans = KMeans( init='k-means++', n_clusters=clusters ).fit( data )
    kmeans = KMeans( init='k-means++', n_clusters=clusters )
    kmeans.fit( reduced_data )

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    ax = np.arange( x_min, x_max, h) 
    ay = np.arange( y_min, y_max, h)
    xx, yy = np.meshgrid( ax, ay )

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict( np.c_[xx.ravel(), yy.ravel()] )

    centroids = kmeans.cluster_centers_             # Put the result into a color plot
    Z = Z.reshape( xx.shape)

    plt.figure( 1)       # new figure managed in some black box space
    plt.clf()           # clear figure
    plt.imshow( Z, interpolation='nearest',
               extent=( xx.min(), xx.max(), yy.min(), yy.max() ),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot( reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=4 )
    centroids = kmeans.cluster_centers_                         # Plot the centroids as a white X
    plt.scatter( centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)

    plt.title( title )
    plt.xlim( x_min, x_max )
    plt.ylim( y_min, y_max )
    plt.xticks(())
    plt.yticks(())
    plt.savefig( fname, format="eps" )


# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
# if you programme in go, then you recognise the beauty here :) 
flags = {                              # define possible flags and the default: map key, type, default value
    "-c": ("n-class", int_flag, 2),                 # number of class values (default binary)
    "-d": ("output-dir", str_flag, "/tmp"),
    "-f": ("classfirst", bool_flag, False),
    "-i": ("iterations", int_flag, 10),
    "-k": ("k-clusters", int_flag, 2),              # number of clusters to divide into
    "-r": ("random-init", str_flag, "k-means++"),    # use kmeans++ as initialisation by default
    "-s": ("plot-samp-rate", int_flag, 10)          # to keep plot output sizes reasonable, sample at x% for plots
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file testing-file" )

if pparms == None  or  len( pparms ) < 2 :
    printf( "missing filenames on command line (training testing)\n" )
    sys.exit( 1 )

train_fn = pparms[0]              # file names; training validation (test)
test_fn = pparms[1];

clusters = opts["k-clusters"]   # stuff options into more managable vars
nclass = opts["n-class"]
iters = opts["iterations"]
init_str = opts["random-init"]
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
    vdata = test_data.iloc[:,1:train_p]
    vlabels = test_data.values[:,0]               # get just the first column which has 'truth'
else :
    data = train_data.iloc[:,0:train_p-1]
    labels = train_data.values[:,-1]               # get just the last column which has 'truth'
    vdata = test_data.iloc[:,0:train_p-1]
    vlabels = test_data.values[:,-1]               # get just the first column which has 'truth'

data = data.values
data_n, data_p = data.shape          # data instances and parameters
printf( "#data: %d instances  %d parameters\n", data_n, data_p )

printf( "#%6s %-6s %-6s %-6s %-6s %-6s %-6s %-10s %s\n", "N-clust", "ACC", "HOMO", "COMPL", "VM", "ARAND", "MI", "CH-score", "iters" )

for i in range( iters ) :
    kmeans = KMeans( init=init_str, n_clusters=clusters, max_iter=100, n_init=i+1 )   # arrange in clusters (non-deterministic if iters > 1)
    kmeans.fit( data )
    #kmeans = KMeans( n_clusters=clusters, init="random",algorithm="elkan",  max_iter=800  ).fit( data ) 
    guess = kmeans.predict( data )
    #labels = labels
    #guess = kmeans.labels_

    acc = metrics.accuracy_score( labels, guess )
    homo = metrics.homogeneity_score( labels, guess )         # generate and print other stats
    comp = metrics.completeness_score( labels, guess )
    vm = metrics.v_measure_score( labels, guess ) 
    arand = metrics.adjusted_rand_score( labels, guess )
    mi = metrics.adjusted_mutual_info_score( labels,  guess, average_method="arithmetic" )
    sil = metrics.silhouette_score( data, guess, metric='euclidean', sample_size=4800 )
    ch = metrics.calinski_harabaz_score( data, guess );

    printf( " %6d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %10.3f %d\n", clusters, acc, homo, comp, vm, arand, mi, ch, kmeans.n_iter_ )

    if i == 0 :                                     # plot only the first with a consistent n-init
        tokens = train_fn.split( "/" );             # build file name as kmeans_<data-type>_<clusters>.eps
        tokens = tokens[-1].split( "_" )
        title = sprintf( "K-Means %s k=%d", tokens[0], clusters ) 
        gfname = sprintf( "%s/kmeans_%s_%d.eps", out_dir, tokens[0], clusters  )
        km_cplot( data, clusters, gfname, title, sample_pct=opts["plot-samp-rate"],rs=12 )    # colour markers plot

        gfname = sprintf( "%s/kmeans_%s_gt.eps", out_dir, tokens[0] )
        gt_cplot( data, labels, nclass, gfname, title, sample_pct=opts["plot-samp-rate"],rs=12 )    # ground truth plot

sys.exit( 0 )


