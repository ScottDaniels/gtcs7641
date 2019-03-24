#   Mnemonic:   dimred.py
#   Abstract:   Dimension reduction over a dataset
#
#   Author:     E. Scott Danies
#   Date:       05 March 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#       code examples from the following URLs:
#           https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#           https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
#           Variable names changed to protect the innocent, and make more sense IMHO.
#
#       Plot suggestions without a display attached
#           https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
#
# ------------------------------------------------------------------

import matplotlib
matplotlib.use('Agg')      # because this makes all kinds of sense; must be called before the pyplot import! WTF python
import matplotlib.pyplot as plt

import math
import numpy as np
import pandas as pd

import sklearn
from sklearn.cluster import KMeans
#from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
#from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.decomposition import FastICA


import sys
# how lame is python that this needs to actually be coded by the user?
def printf( format, *args ) :
    sys.stdout.write( format % (args) )
    sys.stdout.flush()



#
#   determine the point pair in points that x,y is closest to
#
def find_closest( x, y, points ) :
    mind = 0
    for p in range( len(points) ) :
        pp = points[p]
        px = pp[0]
        py = pp[1]
        #px = points[p,0]
        #py = points[p,1]
        #print( ">>>", points, len(points), p )
        d = math.sqrt( ((x-px)**2) + ((y-py)**2) )
        if p == 0 or d < mind :
            mind = d
            i = p

    return i
        
# this is a hack from code from the url:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
#
def get_centers( gmm, X ) :
    centroids = []
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        #foo = []
        #foo.append( pos[0] )
        #foo.append( pos[1] )
        centroids.append( [pos[0], pos[1]] )

    return centroids

        
#
#   This is an alternative to the kmeans plotter in the sklearn examples which generate HUGE
#   and thus unusable .esp files.
#   Plot with coloured points with one colour/style for each cluster; max 20. To reduce the number
#   of points in the plot, and thus the size, the data is sampled (default 10%). sample_pct can be
#   inccreased to include more; this is pretty course grained because of python's step limits and 
#   my unwillingness to spend more time on this.
#
#   The random state parm (rs) default of 12 seems to yield the best distribution for plotting.
def km_cplot( data, clusters, fname, title="", sample_pct=20, algo="grp", rs=12 ) :
    if algo == "pca" :
        plot_data = PCA( n_components=2 ).fit_transform( data )     # reduce to 2d for plotting
    else :
        plot_data = GaussianRandomProjection( n_components=2, random_state=rs ).fit_transform( data )   # reduce to 2d for plot

    kmeans = KMeans( init='k-means++', n_clusters=clusters, n_init=19 )     # arrange in clusters
    #kmeans = KMeans( init='k-means++', n_clusters=clusters, n_init=10 )     # arrange in clusters
    kmeans.fit( plot_data )                                                 # compute center of each cluster

    #x_min, x_max = plot_data[:, 0].min() - 1, plot_data[:, 0].max() + 1
    #y_min, y_max = plot_data[:, 1].min() - 1, plot_data[:, 1].max() + 1
    x_min = plot_data[:, 0].min() - 1       # plot dimensions
    x_max = plot_data[:, 0].max() + 1
    y_min = plot_data[:, 1].min() - 1 
    y_max = plot_data[:, 1].max() + 1

    plt.figure( 1 )       # new figure managed in some black box space
    plt.clf()           # clear figure

    rows, columns = plot_data.shape
    gap = int( 100/sample_pct  )                    # sample data and plot an nth to keep output smaller
    centroids = kmeans.cluster_centers_             # center of each cluster
    #printf( ">>> %s\n", centroids )

    totals = []
    for i in range( clusters ) :
        totals.append( 0 ) 

    colours = [ "r+", "g+", "b+", "m+", "c+", 
                "rD", "gD", "bD", "mD", "cD", 
                "r.", "g.", "b.", "m.", "c.", 
                "r*", "g*", "b*", "m*", "c*", 
                ]    # plotter output colours based on cluster number

    for i in range( rows ) :            # sample the data, and capture if a part of this cluster
        lab = find_closest( plot_data[i,0], plot_data[i,1], centroids ) 
        totals[lab] += 1

    for k in range( clusters ) :                    # for each cluster; find points closest to its center
        sampled_data_x = []                         # capture points in the cluster, then plot after loop with given colour
        sampled_data_y = []

        for i in range( 0, rows, gap ) :            # sample the data, and capture if a part of this cluster
            if find_closest( plot_data[i,0], plot_data[i,1], centroids ) == k :     # in this cluster, capture it
                sampled_data_x.append( plot_data[i,0] )
                sampled_data_y.append( plot_data[i,1] )

        plt.plot( np.asarray( sampled_data_x ), np.asarray( sampled_data_y), colours[k], markersize=4, rasterized=False )

    if clusters > 1 :       # seems silly to mark if just plotting a single cluster
        plt.scatter( centroids[:, 0], centroids[:, 1], marker="p", color="k",  s=169, linewidths=4, zorder=10)

    #printf( "# plot dist: " )
    #for i in range( clusters ) :
    #    printf( "%.2f ", (totals[i]/rows) * 100  )
    #printf( "\n" )

    plt.title( title )
    plt.xlim( x_min, x_max )
    plt.ylim( y_min, y_max )
    plt.xticks(())
    plt.yticks(())
    plt.savefig( fname, format="eps" )


        
#
#   This will plot "ground truth" given a set of data, and a set of labels. The data is reduced
#   to 2 dimensions, and plotted with a colour/shape corresponding to the label (ground truth)
#   value. The same sampling holds as was described by km_cplot. The nclass parameter is the
#   number of different classifications in labels (range 0-n).
#
#   Probably better named as label plotting as we also use this to plot the EM labeled data which
#   isn't ground truth, but that's how it started :) 
#
#   The random state parm (rs) default of 12 seems to yield the best distribution for plotting.
def gt_cplot( data, labels, nclass, fname, title="", sample_pct=20, algo="grp", rs=12, cpattern=1 ) :
    if algo == "pca" :
        plot_data = PCA( n_components=2 ).fit_transform( data )     # reduce to 2d for plotting (PCA is funky)
    else :
        plot_data = GaussianRandomProjection( n_components=2, random_state=rs ).fit_transform( data )   # reduce to 2d for plot

    x_min = plot_data[:, 0].min() - 1       # plot dimensions
    x_max = plot_data[:, 0].max() + 1
    y_min = plot_data[:, 1].min() - 1 
    y_max = plot_data[:, 1].max() + 1

    plt.figure( 1 )       # new figure managed in some black box space
    plt.clf()           # clear figure

    rows, columns = plot_data.shape
    gap = int( 100/sample_pct  )                    # sample data and plot an nth to keep output smaller

    if cpattern == 1 :
        colours = [ "rD", "gD", "bD", "kD", "mD", 
                "r+", "c+", "bD", "mD", "cD", 
                "r.", "g.", "b.", "m.", "c.", 
                "r*", "g*", "b*", "m*", "c*", 
                ]    # plotter output colours based on cluster number
    else :
        # this set matches the c_cplot function
        colours = [ "r+", "g+", "b+", "m+", "c+", 
                "rD", "gD", "bD", "mD", "cD", 
                "r.", "g.", "b.", "m.", "c.", 
                "r*", "g*", "b*", "m*", "c*", 
                ]    # plotter output colours based on cluster number

    for c in range( nclass ) :                    # for each class print it's members
        sampled_data_x = []                         # capture points in the cluster, then plot after loop with given colour
        sampled_data_y = []

        for i in range( 0, rows, gap ) :            # sample the data, and capture if it has the current label value
            if labels[i] == c :
                sampled_data_x.append( plot_data[i,0] )
                sampled_data_y.append( plot_data[i,1] )

        plt.plot( np.asarray( sampled_data_x ), np.asarray( sampled_data_y), colours[c], markersize=3, rasterized=False )

    plt.title( title )
    plt.xlim( x_min, x_max )
    plt.ylim( y_min, y_max )
    plt.xticks(())
    plt.yticks(())
    plt.savefig( fname, format="eps" )

# -------------
def c_cplot( data, clusters, centroids, fname, title="", sample_pct=20, algo="grp", rs=12 ) :
    if algo == "pca" :
        plot_data = PCA( n_components=2 ).fit_transform( data )     # reduce to 2d for plotting
    else :
        plot_data = GaussianRandomProjection( n_components=2, random_state=rs ).fit_transform( data )   # reduce to 2d for plot

    x_min = plot_data[:, 0].min() - 1       # plot dimensions
    x_max = plot_data[:, 0].max() + 1
    y_min = plot_data[:, 1].min() - 1 
    y_max = plot_data[:, 1].max() + 1

    plt.figure( 1 )       # new figure managed in some black box space
    plt.clf()           # clear figure

    rows, columns = plot_data.shape
    gap = int( 100/sample_pct  )                    # sample data and plot an nth to keep output smaller

    colours = [ "r+", "g+", "b+", "m+", "c+", 
                "rD", "gD", "bD", "mD", "cD", 
                "r.", "g.", "b.", "m.", "c.", 
                "r*", "g*", "b*", "m*", "c*", 
                ]    # plotter output colours based on cluster number

    for k in range( clusters ) :                    # for each cluster; find points closest to its center
        sampled_data_x = []                         # capture points in the cluster, then plot after loop with given colour
        sampled_data_y = []

        for i in range( 0, rows, gap ) :            # sample the data, and capture if a part of this cluster
            if find_closest( plot_data[i,0], plot_data[i,1], centroids ) == k :     # in this cluster, capture it
                sampled_data_x.append( plot_data[i,0] )
                sampled_data_y.append( plot_data[i,1] )

        plt.plot( np.asarray( sampled_data_x ), np.asarray( sampled_data_y), colours[k], markersize=4, rasterized=False )

    #if clusters > 1 :       # seems silly to mark if just plotting a single cluster
    #   for p in range( len(centroids ) ) :
    #       pp = centroids[p]
    #       px = pp[0]
    #       py = pp[1]
    #       plt.scatter( px, py, marker="p", color="k",  s=169, linewidths=4, zorder=10)

    #    plt.scatter( centroids[:, 0], centroids[:, 1], marker="p", color="k",  s=169, linewidths=4, zorder=10)

    plt.title( title )
    plt.xlim( x_min, x_max )
    plt.ylim( y_min, y_max )
    plt.xticks(())
    plt.yticks(())
    plt.savefig( fname, format="eps" )

