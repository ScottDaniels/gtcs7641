
#   Mnemonic:   knn.py
#   Abstract:   Build a k-nearest neighbours classifier and generate performance
#               statistics
#   Author:     E. Scott Danies
#   Date:       17 January 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from, or 
#        code used in some manner from the following URLs:
#           Basic flow using sklearn:
#           http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
#
#           Cross Validation (skorg)
#           https://scikit-learn.org/stable/modules/cross_validation.html

import sys
from time import time

# imports from dataaspirant -- not all worked
#import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
#from sklearn import tree

# imports from JonTay's code -- needed 2 from above to use examples from the dataaspirant page
import sklearn.model_selection as ms
import pandas as pd
#from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  

from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as dtclf

from sklearn.model_selection import cross_val_score

# -- my tools --------------------------------------------------------
from tools import printf,fprintf
from tools import parse_args,bool_flag,val_flag,int_flag,str_flag
from tools import print_pd
from tools import mk_lc_data, print_lc_data


# ----------------------------------------------------------------------------------------------------------
flags = {                              # define possible flags and the default: map key, type, default value
    "-a": ("pr_ave", str_flag, "binary" ),
    "-N": ("normalise", bool_flag, False),      # this seems to hurt dispite what stack abuse says!
    "-f": ("classfirst", bool_flag, False),
    "-i": ("iterations", int_flag, 0),          # if set > 0, then we subtract from max-neighbours and use as starting point
    "-l": ("learn_curve", bool_flag, True),     # -l turns off
    "-n": ("max_neighbours", int_flag, 10),
    "-s": ("rstate", int_flag, 100), 
    "-t": ("tsize", int_flag, 30) 
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if len( pparms ) < 1 :
    printf( "missing filename on command line\n" )
    sys.exit( 1 )

filen = pparms[0]
tsize = opts["tsize"]/100            # size of the raw data to resrve for validation (pct)
rstate = opts["rstate"]              # random state value
iterations = opts["iterations"]      # random state value
gen_lc = opts["learn_curve"]
pr_ave = opts["pr_ave"]              # precision/recall average method
max_neighbours = opts["max_neighbours"] 
if len( pparms ) > 1 :              # a second filename assumed to be a separate test data set for validation
    vfilen = pparms[1]
    have_val_data = True
else :
    have_val_data = False

# ----- load and split the data ----------------------------------------------------------

raw_data = pd.read_csv( filen, sep= ',' )

#print( "Dataset Lenght:: ", len( raw_data) )
#print( "Dataset Shape:: ",  raw_data.shape )
dheight, dwidth = raw_data.shape


if opts["classfirst"] :                   # class target value is in col 0
    X = raw_data.values[:, 1:dwidth-1]    # attributes are field 1 through n
    Y = raw_data.values[:, 0]             # the truth value is in the 0th field
else :
    X = raw_data.values[:, 0:-2]          # attributes are 0 through n-1
    Y = raw_data.values[:,dwidth-1]       # the truth value is in the nth field

x_train, x_test, y_train, y_test = ms.train_test_split( X, Y, test_size=tsize, random_state=rstate ) # split data into training and test; 30% for testing
trh, trw = x_train.shape

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

# stack abuse page suggests feature scaling to prevent an attribute with a bigger range
# from dominating the contribution to the generated function. The following does this
# unless -N was given on command line.  This seems not to be the case, default is off
if opts["normalise"] :
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform( x_train )  
    x_test = scaler.transform( x_test )  


# ------------------ build model, predict, measure ---------------------------
if iterations > 0 and iterations < max_neighbours :
    neighbours = (max_neighbours - iterations) + 1
else :
    neighbours = 1

accuracy = []              # accuracy for each iteration using reserved data
vaccuracy = []             # accuracy for each iteratoin using provided validation data
cv_info = []               # average cross validation score for each neighbour value
prec = []                   # precision and recall stats
recall = []
elapsed = []
pred_elapsed = []           # time required for prediction
samples = []
ncount = []

if gen_lc :
    cfier = KNeighborsClassifier( n_neighbors=neighbours )
    lc_data = mk_lc_data( cfier, X, Y )                     # compute the learning curve; just once
    print_lc_data( lc_data, tot_instances=dheight )

while neighbours <= max_neighbours :
    #fprintf( sys.stderr, "computing with neighbours = %d\n", neighbours )
    samples.append( trh )
    ncount.append( neighbours )

    cfier = KNeighborsClassifier( n_neighbors=neighbours )

    cvscores = cross_val_score( cfier, X, Y, cv=5 )
    sum = 0.0
    for s in cvscores :
        sum += s
    cv_ave = (sum/len( cvscores )) * 100
    cv_info.append( cv_ave )
    

    start_ts = time()
    cfier.fit( x_train, y_train )             # build the tree using the training data
    end_ts = time()
    elapsed.append( end_ts - start_ts )

    start_ts = time()
    predicted_y = cfier.predict( x_test )                               # predict using model
    end_ts = time()
    pred_elapsed.append( end_ts - start_ts )
    accuracy.append( accuracy_score( y_test, predicted_y ) * 100 )      # suss out accuracy using pruned tree
    if have_val_data :
        predicted_y = cfier.predict( vx_test )                          # use model to predict the blind test data
        vaccuracy.append( accuracy_score( vy_test, predicted_y) * 100 )
        prec.append( precision_score( vy_test, predicted_y, average=pr_ave ) )        # measure of accuracy (higher means higher overall accuracy)
        recall.append( recall_score( vy_test, predicted_y, average=pr_ave ) )         # measure of usefullness (higher means less positive misses)

    neighbours += 1

if have_val_data :
    printf( "#---snip accuracy stats -----\n" )
    printf( "#\tngbrs   elapsed pelapsd rsvd    test    cv_scor precsn  recall samples\n" )
    print_pd( [ ncount, elapsed, pred_elapsed, accuracy, vaccuracy, cv_info, prec, recall, samples ], "\t%.3f ", False, 1 )
else :
    printf( "#neigh reserved  cv_score\n" )
    print_pd( [ accuracy, cv_info ], "\t%.3f ", True, 1 ) 

