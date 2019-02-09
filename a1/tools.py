
#   Mnemonic:   tools.py
#   Abstract:   Tools common to all of the needed code.
#               With the exception of the learning curve function which
#               is based on code/information described at the HTML pgage
#               incidcated in the funciton, all of this code is original.
#   Author:     E. Scott Daniels edaniels@gatech.edu
#   Date:       15 January 2019
# ---------------------------------------------------------------

import sys

import numpy as np
import sklearn.model_selection as ms

# imports from sklearn site -- not all worked
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score

#import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectFromModel
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler

#from collections import defaultdict
#from sklearn.metrics import make_scorer, accuracy_score
#from sklearn.utils import compute_sample_weight
#from sklearn.tree import DecisionTreeClassifier as dtclf

from sklearn.model_selection import cross_val_score

# -------------------------------------------------------------------

# how lame is python that this needs to actually be coded by the user?
def printf( format, *args ) :
    sys.stdout.write( format % (args) )
    sys.stdout.flush()

# print to some device
def fprintf( target, format, *args ) :
    target.write( format % (args) )
    target.flush()

def sprintf( format, *args ) :
    return format % args

# A more realistic flag parser than seems to exist.
#
# Caller should define 'recognised' to be a map of "flag": var-data where var-data
# is a tuple ("varname", bool-type, default value). If bool-type is True, then we
# expect -x and nothing more on the command line. If bool-type is False, then we 
# expect -x value. When -x  is encountered, we set vname in the values map to true 
# (if boolean) or to the value found as the next parameter.  The only thing thig 
# does not support is the very old school style of -cfi filename foo where 
# filename is associated with the 'f' in -cfi and foo is associated with the 'i'.   
# Tar may support this;  it was one of the few Unix mistakes, so this code does not.
#
# The return value is an array of the positional  parameters following the flags and
# values is filled in with either the defaults supplied in recognised, or what 
# we found to override them on the command line. 
#
# Default values can/should be set in the values
# map before passing to this function.
#

bool_flag = "b"       # flag is a boolean type which toggles the default to the opposite
val_flag = "v"        # flag has a value (either -f=value or -f value) and value is float
int_flag = "i"        # flag has a value of int type 
str_flag = "s"        # has a value of string type

def parse_args( recognised, values, pos_help="" ) :
    a = sys.argv
    abort = False

    for fname in recognised :                   # insert default values
        flag, ftype, def_value = recognised[fname]
        values[flag] = def_value

    if len( a ) < 2 : 
        return None

    i = 1
    pp_array = None
    while i < len( a ) :
        parm = a[i]
        if parm[0] != "-" :                     # first token without lead dash is a positional parm
            pp_array = a[i:]
            break

        if parm == "--" :                       # special -- to allow positional parms with lead dashes
            if i+1 < len( a ) :                 # ignore and return
                pp_array = a[i+1:]
            break

        if parm == "-?" or parm == "--help" :   # these special flags shouldn't be defined by caller
            printf( "usage: %s [options] %s\n  options:\n", a[0], pos_help )
            flags = []
            for k in recognised :
                flag, ftype, def_value = recognised[k]
                if ftype == bool_flag :
                    if def_value :
                        tf = "true"
                    else :
                        tf = "false"
                    flags.append(  sprintf( "%s %s (%s)", k, flag, tf ) )
                else :
                    if ftype == val_flag :
                        flags.append( sprintf( "%s n %s (%d)", k, flag, def_value ) )
                    else :
                        flags.append( sprintf( "%s n %s (%s)", k, flag, def_value ) )

            sflags = sorted( flags )
            for e in sflags :
                printf( "\t%s\n", e )
            sys.exit( 0 )
                
        tokens = parm.split( "=", 2 )          # if --foo=bar (-f=bar for that matter)
        if len( tokens ) > 1 :
            parm = tokens[0]
            
        if parm in recognised :
            vname, ftype, dev_val = recognised[parm] 
            if ftype == bool_flag :
                if vname in values :
                    values[vname] = not values[vname]      # flag sets the value to the inverse
                else :
                    values[vname] = True                   # if not set, then this turns on
            else :
                if len( tokens ) > 1 :                      # -f= bar or --foo=bar
                    v = tokens[1]
                else :                                     # either -f bar or --foo bar
                    v = a[i+1]
                    i += 1
                if ftype == val_flag :
                    values[vname] = float( v )
                else :
                    if ftype == int_flag :
                        values[vname] = int( v )
                    else :
                        values[vname] = v
        else :
            abort = True
            printf( "[FAIL] unrecognised flag: %s\n", parm )

        i += 1

    if abort :
        sys.exit( 1 )

    return pp_array
        

# print data for plot_data style plotters.
# One column per line. Data is assumed to be an array of
# data arrays. If index is true, then a 0th column which is
# the index into the data array is printed adjusted by the
# value of index_offset.
#
# If the arrays in data are not all of the same length, then
# the shortest one is used to gate the output.
#
def print_pd( data, fmt="%d ", index=False, index_offset=0 ) :
    if len( data ) < 1 :
        printf( "no data to print: %s\n", data )
        return

    max_index = len( data[0] )
    for i in range( 1, len( data ) ) :
        if len( data[i] ) < max_index :
            max_index = len( data[i] )
    
    for i in range( max_index ) :
        if index :
            printf( "%d ", i + index_offset )
        for j in range( len( data ) ) :
            printf( fmt, data[j][i] )

        printf( "\n" )
    

# ------------ learning curve (irtchieng) -------------
#
#   Compute the learning curve for the classifier (which has a fit() function)
#   using the data x, and y values.  Returns an array of tuples: ( nexamp, mtrs, mts )
#   where nexamp is the number of training samples used, mtrs is the mean training
#   score, and mts is the mean testng score. If gen_error is set to true, then
#   the resulting output is error values and not acuracy.
#   We assume 2 jobs can be run in parallel.
#   
#   This following HTML site contributed the needed information to write the
#   call to the learning curve function and capture mean values:
#           https://www.ritchieng.com/machinelearning-learning-curve/
#
def mk_lc_data( clfier, X, y, gen_error=False, folds=5, samples=11, shuffle=True ) :
    lc_sample_spec = np.linspace( .1, 1.0, samples )     # training sample specification; split to geneate n values

    train_sizes, train_scores, test_scores = ms.learning_curve( clfier, X, y, random_state=17, shuffle=shuffle, n_jobs=3, cv=folds, train_sizes=lc_sample_spec, verbose=20)
    #train_sizes, train_scores, test_scores = ms.learning_curve( clfier, X, y, train_sizes=lc_sample_spec, n_jobs=3, verbose=20)

    train_mean = np.mean( train_scores, axis=1 )
    test_mean = np.mean( test_scores, axis=1 )

    #train_std = np.std(train_scores, axis=1)           # standard deviation not used.
    #test_std = np.std(test_scores, axis=1)

    rv = []
    for i in range( len( train_sizes ) ) :              # build return array of data
        if gen_error :
            rv.append(  (train_sizes[i], 1.0 - train_mean[i], 1.0 - test_mean[i] ) )
        else :
            rv.append(  (train_sizes[i], train_mean[i], test_mean[i] ) )

    return rv

    if False :
        printf( "\n===== lc info =====\n" );
        printf( "sizes %s\n", train_sizes );
        #printf( "train scores %s\n", train_scores );
        printf( "train scores mean %s\n", train_scores_mean );
        #printf( "test scores %s\n", test_scores );
        printf( "test scores mean %s\n", test_scores_mean );
        printf( "\n" )

#
#   Given an array of data from mk_lc_data, print it in plotdata format. If raw is true
#   then we don't multiply it up by 100.
#
def print_lc_data( lc_data, raw=False, tot_instances=0 ) :
    printf( "#---snip learning curve stats -----\n" )
    printf( "#sklearn generated learning curve values\n#T-acc is training accuracy; V-acc is validation accuracy\n" )
    printf( "#\tT-samp  pct T-acc   V-acc\n" )
    for d in lc_data :
        if tot_instances > 0 :
            pi = (d[0] / tot_instances) * 100
        else :
            pi = 0

        if raw :
            printf( "\t%6d  %3d %6.3f %6.3f\n", d[0], pi, d[1], d[2] )
        else :
            printf( "\t%6d  %3d %6.3f %6.3f\n", d[0], pi, d[1]*100, d[2]*100 )
    
