
#   Mnemonic:   mpl.py
#   Abstract:   Run an MLP classifier (neuro-net) on the provided data.
#   Author:     E. Scott Danies
#   Date:       2 February 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#        code used in some manner from the following URLs:
#           Basic flow using sklearn:
#               https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised


import sys
from time import time

# imports from sklearn site -- not all worked
import pandas as pd
import numpy as np
import sklearn.model_selection as ms

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.model_selection import cross_val_score

#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#from collections import defaultdict
#from sklearn.metrics import make_scorer, accuracy_score
#from sklearn.utils import compute_sample_weight
#from sklearn.tree import DecisionTreeClassifier as dtclf


# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf
from tools import parse_args,bool_flag,val_flag,str_flag,int_flag
from tools import print_pd
from tools import mk_lc_data, print_lc_data



# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
flags = {                              # define possible flags and the default: map key, type, default value
    "-a": ("pr_ave", str_flag, "binary" ),
    "-N": ("normalise", bool_flag, True),
    "-f": ("classfirst", bool_flag, False),
    "-g": ("info_gain", bool_flag, True),     # turn off which sets gini
    "-i": ("iterations", int_flag, 50),       # max value for iterations (values selected from the step pool)
    "-l": ("learn_curve", bool_flag, True),   # output should be learning curve info
    "-r": ("learn_rate", str_flag, "constant"), # must be one of constant, adaptive, invscaling
    "-s": ("rstate", int_flag, 100), 
    "-S": ("solver", str_flag, "adam"),         # solver (must be adam, lbfgs, or sgd)
    "-t": ("tsize", val_flag, 30) 
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 1 :
    printf( "missing filename on command line\n" )
    sys.exit( 1 )

filen = pparms[0]                       # raw data file manditory first parameter
gen_lc = opts["learn_curve"]
pr_ave = opts["pr_ave"]
tsize = opts["tsize"] /100        # size of the raw data to resrve for validation (pct)
max_iter = opts["iterations"]
rstate = opts["rstate"]           # random state value

if len( pparms ) > 1 :              # a second filename assumed to be a separate test data set for validation
    vfilen = pparms[1]
    have_val_data = True
else :
    have_val_data = False


# ----- load and split the data ----------------------------------------------------------

raw_data = pd.read_csv( filen, sep= ',' )

#print( "Dataset Lenght:: ", len( raw_data) )
#print( "Dataset Shape:: ",  raw_data.shape )
dheight, dwidth = raw_data.shape          # dheight -- total number of instances in training set


if opts["classfirst"] :                   # class target value is in col 0
    X = raw_data.values[:, 1:dwidth-1]    # attributes are field 1 through n
    Y = raw_data.values[:, 0]             # the truth value is in the 0th field
else :
    X = raw_data.values[:, 0:-2]          # attributes are 0 through n-1
    Y = raw_data.values[:,dwidth-1]       # the truth value is in the nth field

x_train, x_test, y_train, y_test = ms.train_test_split( X, Y, test_size=tsize, random_state=rstate ) # split data into training and test; 30% for testing
train_h, train_w = x_train.shape
fprintf( sys.stderr, "#training set entries: %d\n", train_h )

if have_val_data :                              # a validation/test data set provided; read and spit it out
    val_data = pd.read_csv( vfilen, sep="," )
    vdheight, vdwidth = val_data.shape
    verror = []
    if opts["classfirst"] :                     # target class is in col 0
        vX = val_data.values[:, 1:vdwidth-1]    # attributes are 1 through n
        vY = val_data.values[: ,0]              # the truth value is in col 0
    else :                                      # target class is in nth col
        vX = val_data.values[:, 0:vdwidth-2]    # attributes are 0 through n-1
        vY = val_data.values[:, vdwidth-1]      # the truth value is in the nth field

    vx_train, vx_test, vy_train, vy_test = ms.train_test_split( vX, vY, test_size=100, random_state=rstate )    # 'reserve' all for in the test


# ----- normalise training data if option is set ------------------------------------------

# The sklearn doc on the MLPC classifier indicates that normalisation of data is 
# a requirement.
#
if opts["normalise"] :          # it can be disbled from the command line for experimentation (-N)
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform( x_train )  
    x_test = scaler.transform( x_test )  
    vx_train = scaler.transform( vx_train )  
    vx_test = scaler.transform( vx_test )  


# ------------------ build trees, predict, measure ---------------------------
niters = []                 # number of iterations max at each run
accuracy = []              # accuracy stats -- unpruned
vaccuracy = []             # unpruned on validation set
cv_info = []               # average cross validation score for each depth
prec = []                  # precision and recall
recall = []
elapsed = []               # time it takes to fit the data
tinstances = []                  # size of the training set (static, but we need an array to match with other data)


if gen_lc :              # gen a learning curve (takes about 5 min for 11 samples)
    #fprintf( sys.stderr, "generating learning curve..." )
    cfier = MLPClassifier( solver=opts["solver"] )
    lc_data = mk_lc_data( cfier, X, Y, samples=11 )     # compute the learning curve (smaller set of samples)
    print_lc_data( lc_data, tot_instances=train_h )

iter_pool = [ 10, 50, 100, 200, 400, 800 ]      # number of iterations to set in the classifier (default is 200)

iidx = 0
while iidx < len( iter_pool )  and  iter_pool[iidx] <= max_iter :               # collect stats on how the number of estimators changes things
    tinstances.append( train_h )
    iters = iter_pool[iidx]
    #printf( "computing with max iters == %d\n", iters )
    niters.append( iters )
    cfier = MLPClassifier( validation_fraction=tsize, max_iter=iters, learning_rate=opts["learn_rate"], solver=opts["solver"]  )

    cvscores = cross_val_score( cfier, X, Y, cv=5 )     # generate cross validation score for the depth
    sum = 0.0
    for s in cvscores :
        sum += s
    cv_ave = (sum/len( cvscores )) * 100
    cv_info.append( cv_ave )
    
    ts_start = time()
    cfier.fit( x_train, y_train )             # build the model with the training data
    ts_end = time()
    elapsed.append( ts_end - ts_start )

    predicted_y = cfier.predict( x_test )                               # preduct using test data
    accuracy.append( accuracy_score( y_test, predicted_y ) * 100 )      # capture accuracy of the model on reserved training data
    if have_val_data :
        predicted_y = cfier.predict( vx_test )                          # run and predict based on unseen test (validation) data
        vaccuracy.append( accuracy_score( vy_test, predicted_y) * 100 ) # capture this accuracy
        prec.append( precision_score( vy_test, predicted_y, average=pr_ave ) )        # measure of accuracy (higher means higher overall accuracy)
        recall.append( recall_score( vy_test, predicted_y, average=pr_ave ) )         # measure of usefullness (higher means less positive misses)

    iidx += 1

# print either the manually computed learning curve values or the overall stats/depth
printf( "#---snip accuracy stats -----\n" )
if have_val_data :
    printf( "#\titers   elapsed train   test    cv_sco  precsn  recall T-size\n" )
    print_pd( [ niters, elapsed, accuracy, vaccuracy, cv_info, prec, recall, tinstances ], "\t%-6.2f ", False, 0 )
else :
    printf( "#depth elapsd train   cv_score\n" )
    print_pd( [ niters, elapsed, accuracy, cv_info  ], "\t%.2f ", False,  0 )

