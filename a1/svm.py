#   Mnemonic:   svm.py
#   Abstract:   Implements a SVM learner.
#   Author:     E. Scott Danies
#   Date:       28 January 2019
#
#   Acknowledgements: 
#       This code is based in part on information gleaned from or code snipits
#       from these URLs:
#           https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/
#           https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
#           https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

import sys
from time import time

import sklearn.model_selection as ms
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as dtclf

from sklearn.model_selection import cross_val_score

# ---- my tools -----------------------------------------------------------
from tools import printf,fprintf
from tools import parse_args,bool_flag,val_flag,int_flag,str_flag
from tools import print_pd
from tools import mk_lc_data, print_lc_data

        
# ----------------------------------------------------------------------------------------------------------

valid_ktypes = { "linear": True, "poly": True, "rbf": True }      # easy check to validate -k option

# -- parse command line and convert to convenience variables -----------------------------------------------
flags = {                              # define possible flags and the default: map key, type, default value
    "-a": ("pr_ave", str_flag, "binary" ),
    "-N": ("normalise", bool_flag, True),
    "-f": ("classfirst", bool_flag, False),
    "-k": ("kernel_type", str_flag, "linear"),
    "-l": ("learn_curve", bool_flag, True),    # generate learning curve unless -l given
    "-s": ("rstate", int_flag, 100), 
    "-t": ("tsize", int_flag, 30) 
}
opts = { }                                    # map where option values or defaults come back
pparms = parse_args( flags, opts, "training-file [testing-file]" )

if pparms == None  or  len( pparms ) < 1 :
    printf( "missing filename on command line\n" )
    sys.exit( 1 )

kernel_type = opts["kernel_type"]
if not kernel_type in valid_ktypes :
    printf( "[FAIL] %s is not a valid kernel type for -k parameter\n", kernel_type )
    printf( "\tvalid types are: linear, polynomial, or RBF\n" )
    os.exit( 1 )
    
filen = pparms[0]                       # raw data file manditory first parameter
pr_ave = opts["pr_ave"]
gen_lc = opts["learn_curve"]
tsize = opts["tsize"]/100             # size of the raw data to resrve for validation (pct)
rstate = opts["rstate"]               # random state value
if len( pparms ) > 1 :                # a second filename assumed to be a separate test data set for validation
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

x_train, x_test, y_train, y_test = ms.train_test_split( X, Y, test_size=tsize, random_state=rstate ) # split data into training and test; tsize for testing
print( "train x dataset shape: ",  x_train.shape )
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

# A stack abuse page suggests feature scaling to prevent an attribute with a bigger range
# from dominating the contribution to the generated function. The following does this
# unless -N was given on command line.  
# This makes a HUGE difference for the poly kernel, in terms of execution time, so we 
# default to having it on for this learner. Both training and validation sets must be processed.
#
if opts["normalise"] :
    scaler = StandardScaler()  
    scaler.fit(x_train)
    x_train = scaler.transform( x_train )  
    x_test = scaler.transform( x_test )  
    vx_train = scaler.transform( vx_train )  
    vx_test = scaler.transform( vx_test )  


# ------------------ learn, predict, measure ---------------------------

cfier = SVC( kernel=kernel_type, gamma="auto", verbose=0 )     # force gamma to auto to avoid pesky warning

if gen_lc :                                                    # use sklearn to build the learning curve
    lc_data = mk_lc_data( cfier, X, Y, samples=4, shuffle=True, folds=3 )                        # compute the learning curve
    print_lc_data( lc_data, tot_instances=trh )

#cvscores = cross_val_score( cfier, X, Y, cv=5 )     # generate cross validation score
#sum = 0.0
#for s in cvscores :
#    sum += s
#cv_ave = (sum/len( cvscores )) * 100
#fprintf( sys.stderr, "crossvalidaton captured\n" )


ts_start = time()
cfier.fit( x_train, y_train )             # your going to learn, learn, learnnnnnn
ts_end = time()
elapsed = ts_end - ts_start

predicted_y = cfier.predict( x_test )                               # prediction using the reserved training data
accuracy =  accuracy_score( y_test, predicted_y ) * 100      # suss out accuracy using unpruned tree

if have_val_data :
    predicted_y = cfier.predict( vx_test )                          # predict using the testing set
    vaccuracy = accuracy_score( vy_test, predicted_y) * 100
    prec = -1 #precision_score( vy_test, predicted_y, average=pr_ave )  # measure of accuracy (higher means higher overall accuracy)
    recall = -1 #recall_score( vy_test, predicted_y, average=pr_ave )   # measure of usefullness (higher means less positive misses)

if have_val_data :
    printf( "#---snip accuracy stats -----\n" )
    printf( "#\telapsed train   test    precsn  recall  samples   kernel=%s\n", kernel_type )
    print_pd( [ [elapsed], [accuracy], [vaccuracy], [prec], [recall], [dheight - (tsize * dheight)] ], "\t%.3f ", False, 0 )       # print tools set to take array of arrays; we have just one val each
else :
    printf( "#reserved-train  cv_score\n" )
    print_pd( [ accuracy, cv_info ], "\t%.3f ", True, 0 )
