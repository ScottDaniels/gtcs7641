
#   Mnemonic:   tools.py
#   Abstract:   Tools common to all of the programmes here.
#               Originally written for assignment 1, enhanced for this
#               and duplicated for simplicity (no easy reference to a1
#               things).
#
#   Author:     E. Scott Daniels edaniels@gatech.edu
#   Date:       15 January 2019
# ---------------------------------------------------------------

import sys

import numpy as np
import sklearn.model_selection as ms
from sklearn import metrics

from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

# -------------------------------------------------------------------
from plotter import km_cplot

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
            printf( "usage: %s [options] %s\n  options:   (defaults in parens)\n", a[0], pos_help )
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
                    if ftype == val_flag or ftype == int_flag :
                        flags.append( sprintf( "%s n   --  %s (%d)", k, flag, def_value ) )
                    else :
                        flags.append( sprintf( "%s <str>  -- %s (%s)", k, flag, def_value ) )

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
    
#
#   Run kmeans on the given data for over a range of different n-cluster values.
#   for the desired number of iterations (1 if not given). Data is the data (features)
#   and truth is an array of the ground truth classification of each sample in data.
#
#   The n_init must be set so as to be detrministic which keeps the set of points 
#   the  same across all cluster sizes.
#
def run_kmeans( data, truth, fcount, cvalues, iterations=1, tprefix="", fprefix="reddim", gen_stats=True, method="unk", plot=True ) :
    for cidx in range( len( cvalues ) ) :
        c = cvalues[cidx]                           # number of clusters this round
        for i in range( iterations ) :
            #kmeans = KMeans( n_clusters=c, n_init=1, init="random", algorithm="elkan",  max_iter=100  )
            kmeans = KMeans( n_clusters=c, init="k-means++", n_init=1,  max_iter=100  )
            kmeans.fit( data ) 
            guess = kmeans.predict( data )

            if iterations > 1 or gen_stats :
                acc = metrics.accuracy_score( truth, guess )
                homo = metrics.homogeneity_score( truth, guess )         # compare the true lables to those kmeans predicted
                comp = metrics.completeness_score( truth, guess )
                vm = metrics.v_measure_score( truth, guess ) 
                arand = metrics.adjusted_rand_score( truth, guess )
                mi = metrics.adjusted_mutual_info_score( truth,  guess, average_method="arithmetic" )
                if( c > 1 ) :                                           # ch needs to features minimum, so skip of c == 1
                    ch = metrics.calinski_harabaz_score( data, guess );
                else :
                    ch = 0
        
                printf( " %6s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %10.3f %6d\n", method, acc, homo, comp, vm, arand, mi, ch, c )

        if plot :
            gfname = sprintf( "/tmp/%s_k%d_c%d.eps", fprefix, c, fcount )
            #gfname = sprintf( "/tmp/%s_k%d_c%d.png", fprefix, c, fcount )
            title = sprintf( "%s: f=%d k=%d", tprefix, fcount, c )
            km_cplot( data, c, gfname, title )

        if iterations > 1 :
            printf( "\n" )
        #    km_header( )

