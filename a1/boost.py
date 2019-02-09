
#   Mnemonic:   boost.py
#   Abstract:   Apply adaboost to decision tree as a learner
#   Author:     E. Scott Danies
#   Date:       31 January 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#        code used in some manner from the following URLs:
#           Basic flow using sklearn:
#           http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
#
#           Pruning: (noted inline with soq)
#           https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier/51398390#51398390
#       
#           Pruning scoring, general concept and package references: (noted in line with JonTay)
#           https://github.com/JonathanTay/CS-7641-assignment-1
#
#           Cross Validation (skorg)
#           https://scikit-learn.org/stable/modules/cross_validation.html
#           
#           Boosting
#               https://scikit-learn.org/stable/modules/ensemble.html

import sys
from time import time

import pandas as pd
import numpy as np

#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score

import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler

#from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
#from sklearn.utils import compute_sample_weight
#from sklearn.tree import DecisionTreeClassifier as dtclf

from sklearn.model_selection import cross_val_score

# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf
from tools import parse_args,bool_flag,val_flag,str_flag,int_flag
from tools import print_pd
from tools import mk_lc_data, print_lc_data



# Return true if the node we are examining is a leaf
# (soq)
from sklearn.tree._tree import TREE_LEAF
def is_leaf( inner_tree, index ) :
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

#   Error reduction post prune.  This is based on code from "soq"
#   which provided the initial tree walk code. Extended that base with
#   code fashioned from JonTay's helper functions; my code, his 
#   concept for scoring.
#
#   The tree is traversed, bottom up, and at each node the branch is
#   removed, and the overall accuracy of the tree is evaluated by
#   computing the score on the cross validation set. If the score 
#   better, or the same,  without the branch, then that branch is 
#   pruned.
#
def prune_twigs( tree, model, x_test, y_test, best=0, index=0 ) :
    if not is_leaf( tree,  tree.children_left[index]):
        xbest = prune_twigs( tree, model, x_test, y_test, best, tree.children_left[index] )
        if xbest > best :
            best = xbest
    if not is_leaf( tree,  tree.children_right[index]):
        xbest = prune_twigs( tree, model, x_test, y_test, best, tree.children_right[index] )
        if xbest > best :
            best = xbest

    left = tree.children_left[index]             ## hold original values
    right = tree.children_right[index]
    tree.children_left[index] = TREE_LEAF        # take them out to see if error improves without it
    tree.children_right[index] = TREE_LEAF
    predicted_y = model.predict( x_test )        # predict without the branch
    score = accuracy_score( y_test, predicted_y )
    if score >= best :
        #printf( "pruned: score=%.4f best was %.4f\n", score, best )
        best = score
    else:
        tree.children_left[index] = left        ## put things back as they were
        tree.children_right[index] = right

    return best

# ------------------- print and output support functions -----------------------------------------------
#
#   generate a single line of output with:
#       train-size depth accuracy error
#   Whwere test size is the percent of data used to train. Accuracy and error
#   are selected from the data passed in based on the best cross validation 
#   score.
# 
def print_xlc_data( tsize, depth_base, accuracy, vaccuracy, cv ) :
    max = 0.0
    midx = 0
    for i in range( len( cv ) ) :
        if cv[i] > max :
            max = cv[i]
            midx = i

    printf( "#T-pct  Depth  T-acc  V-acc  V-err\n" )
    printf( "%6d %6d %6.3f %6.3f %6.3f\n", tsize * 100, depth_base + midx, accuracy[midx], vaccuracy[midx], 100 - vaccuracy[midx] )
        
# ----------------------------------------------------------------------------------------------------------

# -- parse command line and convert to convenience variables -----------------------------------------------
flags = {                              # define possible flags and the default: map key, type, default value
    "-a": ("pr_ave", str_flag, "binary" ),
    "-N": ("normalise", bool_flag, False),      # normalise the data first
    "-d": ("depth", int_flag, 25),
    "-e": ("estimators", int_flag, 50),         # max number of estimators (starting with est - iterations)
    "-E": ("est_start", int_flag, 0),         # number of estimators to start with; 0 == beginning of pool
    "-f": ("classfirst", bool_flag, False),
    "-g": ("info_gain", bool_flag, True),     # turn off which sets gini
    "-l": ("learn_curve", bool_flag, True),     # -l turns lc generation off 
    "-p": ("prune", bool_flag, True),         # turn on prune
    "-r": ("learn_rate", val_flag, 1.0 ),     # set the learning rate
    "-s": ("rstate", int_flag, 100), 
    "-t": ("tsize", val_flag, 30) 
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 1 :
    printf( "missing filename on command line\n" )
    sys.exit( 1 )

if opts["info_gain"] :  
    method = "entropy"      # sklearn constant for info gain
else : 
    method = "gini"
    
filen = pparms[0]                       # raw data file manditory first parameter
gen_lc = opts["learn_curve"]
pr_ave = opts["pr_ave"]
tsize = opts["tsize"] /100        # size of the raw data to resrve for validation (pct)
rstate = opts["rstate"]           # random state value
depth = opts["depth"]  
lrate = opts["learn_rate"]  

est_pool = [ 1, 2, 5, 10, 20, 40, 80, 100, 150, 200 ]
max_est = opts["estimators"]  
estart = opts["est_start"]

if len( pparms ) > 1 :              # a second filename assumed to be a separate test data set for validation
    vfilen = pparms[1]
    have_val_data = True
else :
    have_val_data = False

#out_pfx = opts["output_pfx"] + "dtree_" + "method"        # prefix for any output file we open/create

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
txh, txw = x_train.shape

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

# The stack abuse page suggests feature scaling to prevent an attribute with a bigger range
# from dominating the contribution to the generated function. The following does this
# unless -N was given on command line.  At least for the data selected for this project,
# this seems to make minimal difference, so it's off by default.
#
if opts["normalise"] :
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform( x_train )  
    x_test = scaler.transform( x_test )  
    vx_train = scaler.transform( vx_train )  
    vx_test = scaler.transform( vx_test )  


# ------------------ build trees, predict, measure ---------------------------
up_accuracy = []              # accuracy stats -- unpruned
p_accuracy = []               # pruned
up_vaccuracy = []             # unpruned on validation set
p_vaccuracy = []              # pruned on validation set
cv_info = []                  # average cross validation score for each depth
prec = []                       # precision and recall
recall = []
elapsed = []                    # elapsed time to run the fit
samples = []                    # will always be static, but allows easy printing
est_used = []                   # number of estimators this round

# create a base classifer, then prune it so we have two to work from 
up_base_cfier = DecisionTreeClassifier( criterion="gini", random_state=19, max_depth=depth, min_samples_split=10, min_samples_leaf=10 )
up_base_cfier.fit( x_train, y_train )             # build the tree using the training data

base_cfier = DecisionTreeClassifier( criterion="gini", random_state=19, max_depth=depth, min_samples_split=10, min_samples_leaf=10 )
base_cfier.fit( x_train, y_train )             # build the tree using the training data
prune_twigs( base_cfier.tree_, base_cfier, x_test, y_test )       # do post pruning

if gen_lc :              # gen a learning curve; no depth specification for this
    cfier = AdaBoostClassifier( base_estimator=base_cfier, n_estimators=50 )           # compute using the defaults
    lc_data = mk_lc_data( cfier, X, Y )     # compute the learning curve
    print_lc_data( lc_data, tot_instances=dheight )

eidx = 0                                    # index into estimator pool
if estart > 0 :                             # if a desired starting point given, skip to it
    for eidx in range( len( est_pool ) ) :
        if est_pool[eidx] >= estart :
            break

while eidx < len( est_pool )  and  est_pool[eidx] <= max_est :               # collect stats on how the number of estimators changes things
    estimators = est_pool[eidx]
    samples.append( txh )
    est_used.append( estimators )
    #fprintf( sys.stderr, "computing with estimators = %d\n", estimators )

    cfier = AdaBoostClassifier( base_estimator=base_cfier, n_estimators=estimators, learning_rate=lrate, algorithm="SAMME.R", random_state=19 )
    up_cfier = AdaBoostClassifier( base_estimator=up_base_cfier, n_estimators=estimators, learning_rate=lrate, algorithm="SAMME.R", random_state=19 )

    cvscores = cross_val_score( cfier, X, Y, cv=5 )     # generate cross validation score for the depth
    sum = 0.0
    for s in cvscores :
        sum += s
    cv_ave = (sum/len( cvscores )) * 100
    cv_info.append( cv_ave )

    ts_start = time()
    cfier.fit( x_train, y_train )             # build the tree using the training data
    ts_end = time()
    up_cfier.fit( x_train, y_train )             # build the tree using the training data

    elapsed.append( ts_end - ts_start )

    predicted_y = up_cfier.predict( x_test )                            # unpruned prediction
    up_accuracy.append( accuracy_score( y_test, predicted_y ) * 100 )   # suss out accuracy using unpruned tree
    if have_val_data :
        predicted_y = up_cfier.predict( vx_test )                       # unpruned prediction using the validation set
        up_vaccuracy.append( accuracy_score( vy_test, predicted_y) * 100 )

    predicted_y = cfier.predict( x_test )                               # predict based on pruned classifier base
    p_accuracy.append( accuracy_score( y_test, predicted_y ) * 100 )
    if have_val_data :
        predicted_y = cfier.predict( vx_test )       # unpruned prediction using the validation set
        p_vaccuracy.append( accuracy_score( vy_test, predicted_y) * 100 )
        prec.append( precision_score( vy_test, predicted_y, average=pr_ave ) )        # measure of accuracy (higher means higher overall accuracy)
        recall.append( recall_score( vy_test, predicted_y, average=pr_ave ) )         # measure of usefullness (higher means less positive misses)

    eidx += 1


if have_val_data :
    printf( "#---snip accuracy stats -----\n" )
    printf( "#\t\t\t---training---  ------ test ----\n" )
    printf( "#\testmrs  eapsed  unprun  pruned  unprun  pruned  cv_sco  precsn  recall  samples\n" )
    print_pd( [ est_pool, elapsed, up_accuracy, p_accuracy, up_vaccuracy, p_vaccuracy, cv_info, prec, recall, samples ], "\t%6.2f ", False, 0 )
else :
    printf( "       ---training----\n" )
    printf( "#depth unpruned pruned cv_score\n" )
    print_pd( [ est_pool, up_accuracy, p_accuracy, cv_info  ], "\t%.3f ", False, 0 )

