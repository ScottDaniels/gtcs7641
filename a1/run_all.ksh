#!/usr/bin/env bash
# ksh is preferred, but I assume the generic test environment likely not to install it, so 
# bash should be fine here.

#	Mnemonic:	run_all.ksh
#	Abstract:	Run all experiments; should run with bash or ksh.
#	Date:		28 January 2019
#	Author:		Edward Scott Daniels  edaniels7@gatech.edu
#
#	Notes:		crowd data has the target value first (column 0) which 
#				requires the -f option. All experiments are run with a 
#				30% reserve of the training data for validation, and both
#				data sets have a separate test dataset for a test verification.
#				For decision tree, -g enables gini and whem omitted info
#				gain (entrophy) is used.
#
#				This script makes several assumptions as it is taylored spcifically
#				to start jobs on either the adult or crowd data:
#
#					- Data is expected to live in data/crowd and data/adult
#					- The current directory is writable as the output files
#					  will be placed in a subdirectory (run_all_out) unless the
#					  -o option is given to specify a different directory.
#					- the script is being run from a tty/ptty and thus /dev/tty is
#					  available or if -q is given (quiet mode) /dev/null is available.
#					  (*nix is the underlying o/s).
#
#	WARNING:	The generation of learning curve for some classifers requires
#				an insanely long period of time; the polynomial classifier
#				for the Adult data ran for more than 24 hours.  For this
#				reason, generation of learning curve data for the SVM classifer
#				is turned off by default.
# --------------------------------------------------------------------------------------

# Execute the desired command, logging the command to stdout and to the tty.
# Applicatoins are assumed to write all useful information to stdout which is
# captured by the caller.
#
function run {
	last_cmd="$@"

	echo "$@" >$interactive_device		# show progress to user
	if (( ! forreal ))
	then
		return
	fi

	echo "" >>$log			# help identify errors if any are generated (some natter without end)
	echo "$(date)" >>$log
	echo "$@" >>$log

	echo "# $(date)"		# capture date and command used in the output for verification
	echo "# $@"
	#"$@" 2>>$log
	"$@" 
	if (( $? > 0 ))
	then
		echo "[WARN] experiment failed: $@"
		if (( abort_on_fail ))
		then
			exit 1
		fi
	fi
}

function usage {
	echo "usage: $0 [-a] [-l] [-n] [-o output-dir] [-O option] [-t|-T] [-q] [learner-type]"
	cat <<endKat
	-a 	abort on the first failure, otherwise log and continue
	-l	turn on learning curge generation for classifiers that take eons to run them
	-n	no execution mode; say what would be run
	-o	specify the output directory. If not given ./run_all_out is used
	-O	specify parms to be passed directly to each application e.g. -O "-S adam"
		(likely only valid when running for a single classifier)
	-t	enable timing runs and output
	-T	turn off all EXCEPT timing runs
	-q	quiet mode; do not write to tty

	learner type is one of: dtree, boost, knn, mlp or svm. if omitted, then all types
	are assumed and all applications are executed.
endKat
}
# ---------------------------------------------------------------------------------------

abort_on_fail=0			# -a sets and we abort on first failure
gen_timing_output=0
timing_only=0			# -T sets to run only timing
out_d="run_all_out"
force_lc="-l"			# for slugs, turn off lc unless -l is given on OUR cmd line (-l turns off)
forreal=1
interactive_device="/dev/tty"		# use -q (quiet) to turn off (non-interactive mode)

while [[ $1 == "-"* ]]
do
	case $1 in 
		-a)	abort_on_fail=1;;
		-l)	force_lc="";;			# allow lc to run (force on) by removing the -l parm on the slug command lines
		-n) forreal=0;;				# just say, don't do
		-o)	out_d=$2; shift;;
		-O)	opts="$2";shift;;		# any specialised options to give to an execution (e.g. -S adam for mlp to change the solver)
		-t) gen_timing_output=1;;
		-T)	gen_timing_output=1; timing_only=1;;
		-q)	interactive_device="/dev/null";;

		-\?)	usage
				exit 0
				;;

		*)		echo "unrecognised option: $1"
				echo "usage: $0 [-a] [-o output-dir] [-t] [learner-type]"
				exit 1
				;;
	esac
	
	shift
done

if (( ! forreal ))
then
	echo "NOTE: -n given, just listing commands that would be run"
	out_d=./dummy.d
fi

export PYTHONPATH=$PWD/bin:$PYTHONPATH

adult_data="data/adult/adult_parsed.csv data/adult/adult_parsed_test.csv"	# data input parms
crowd_data="data/crowd/crowd_train.csv data/crowd/crowd_test.csv"			
if [[ ! -d ./data ]]		# ensure we can find them
then
	echo "[FAIL] unable to find data direcory in $PWD"
	exit 1
fi

errors=0
for f in $crowd_data $adult_data
do
	if [[ ! -e $f ]]
	then
		echo "[FAIL] cannot find needed input data file: $f"
		errors=1
	fi
done
if (( errors )) 	# report all missing data files, not just first, bail if any not found
then
	exit 1
fi

log=run_all.log
>$log

if ! mkdir -p $out_d		# don't fail because user didn't know to create the directory!
then
	echo "[FAIL] unable to find/create output directory: $out_d"
	exit 1
fi

crowd_data="-f $crowd_data"		# classifier is in first column, so data needs -f

# no parms, run everything, listing parm(s) on the command line selectively runs just those
# suss out what to do from the command line...
if [[ -z $1 ]]
then
	dtree=1
	knn=1
	svm=1
	boost=1
	mlp=1
else
	dtree=0		# run only what matches on the command line
	knn=0
	svm=0
	boost=0
	mlp=0

	while [[ -n $1 ]]
	do
		case $1 in 
			dt|dtree|Dtree)	dtree=1;;
			knn|k-nn|KNN)	knn=1;;
			svm|SVM)		svm=1;;
			boost*|Boost*|dt-bost) boost=1;;
			mlp|ann|neural) mlp=1;;

			*)	echo "[FAIL] unrecognised learner type on command line: $1"
				echo "   use a combination of: dtree, boost, knn, mlp, and/or svm"
				exit 1
				;;
		esac
	
		shift
	done
fi

if (( dtree ))
then
	if (( ! timing_only ))
	then
		run python3 bin/dtree.py       $opts -i 20 -t 30 -d 25  $adult_data >$out_d/adult_dtree_igain.all 
		run python3 bin/dtree.py    -g $opts -i 20 -t 30 -d 25  $adult_data >$out_d/adult_dtree_gini.all 
	
		run python3 bin/dtree.py -a micro    $opts -i 20 -t 30 -d 25  $crowd_data >$out_d/crowd_dtree_igain.all
		run python3 bin/dtree.py -a micro -g $opts -i 20 -t 30 -d 25  $crowd_data >$out_d/crowd_dtree_gini.all
	fi

	if (( gen_timing_output ))
	then
		>$out_d/adult_dtree_timings.all 
		>$out_d/crowd_dtree_timings.all 
		for size in 90 80 70 60 50 40 30 20 10
		do
			run python3 bin/dtree.py  -l $opts $opts -i 1 -t $size -d 25  $adult_data >>$out_d/adult_dtree_timings.all 
			run python3 bin/dtree.py -a micro -g $opts -i 20 -t $size -d 25  $crowd_data >>$out_d/crowd_dtree_timings.all 
		done 
	fi
fi


if (( knn )) 
then
	if (( ! timing_only ))
	then
		run python3 bin/knn.py -n 25  -t 30  $adult_data >$out_d/adult_knn.all 
		run python3 bin/knn.py -n 25  -a micro -t 30  $crowd_data >$out_d/crowd_knn.all 
	fi

	if (( gen_timing_output ))
	then
		>$out_d/crowd_knn_timings.all 
		>$out_d/adult_knn_timings.all 
		for size in 90 80 70 60 50 40 30 20 10
		do
			run python3 bin/knn.py $opts -i 1 -n 9  -l -t $size  $adult_data >>$out_d/adult_knn_timings.all 
			run python3 bin/knn.py -a micro $opts -i 1 -n 9  -l -t $size  $crowd_data >>$out_d/crowd_knn_timings.all 
		done 
	fi
fi

if (( svm ))
then
	for k in linear poly rbf
	do
		if (( ! timing_only ))
		then
			run python3 bin/svm.py  $force_lc -k $k -t 30 $adult_data >$out_d/adult_svm_$k.all 
			run python3 bin/svm.py  $force_lc -k $k -t 30 $crowd_data >$out_d/crowd_svm_$k.all 
		fi

		if (( gen_timing_output ))
		then
			>$out_d/adult_svm_$k_timings.all 
			>$out_d/crowd_svm_$k_timings.all 
			for size in 90 80 70 60 50 40 30 20 10
			do
				run python3 bin/svm.py  -l -k $k -t $size $adult_data >>$out_d/adult_svm_$k_timings.all 
				run python3 bin/svm.py  -l -k $k -t $size $crowd_data >>$out_d/crowd_svm_$k_timings.all 
			done 
		fi
	done
fi

# at 200 evaluators it takes about 15 min to run for crowd
if (( boost ))
then
	if (( ! timing_only ))
	then
		run python3 bin/boost.py  $opts -e 200  -t 30 $adult_data >$out_d/adult_boost.all 
		run python3 bin/boost.py  -a micro $opts -e 200  -t 30 $crowd_data >$out_d/crowd_boost.all 
	fi

	if (( gen_timing_output ))
	then
		>$out_d/adult_boost_timings.all 
		>$out_d/crowd_boost_timings.all 
		for size in 90 80 70 60 50 40 30 20 10
		do
			run python3 bin/boost.py  -l $opts -E 200 -e 200  -t $size $adult_data >>$out_d/adult_boost_timings.all 
			run python3 bin/boost.py  -a micro $opts -E 200 -e 200  -t $size $crowd_data >>$out_d/crowd_boost_timings.all 
		done 
	fi
fi

# This takes about 5min per dataset
if (( mlp ))
then
	if (( ! timing_only ))
	then
		run python3 bin/mlp.py            $opts -i 400 -t 30  $adult_data >$out_d/adult_mlp.all
		run python3 bin/mlp.py   -a micro $opts -i 400 -t 30  $crowd_data >$out_d/crowd_mlp.all
	fi

	if (( gen_timing_output ))
	then
		>$out_d/adult_mlp_timings.all 
		>$out_d/crowd_mlp_timings.all 
		for size in 90 80 70 60 50 40 30 20 10
		do
			run python3 bin/mlp.py       $opts -l -i 400 -t $size  $adult_data >>$out_d/adult_mlp_timings.all 
			run python3 bin/mlp.py   -a micro $opts -i 400 -t $size $crowd_data >>$out_d/crowd_mlp_timings.all 
		done 
	fi
fi

rm -fr ./dummy.d		# if -n given we use this to prevent overlay of last real run, so nix it now
echo "$(date) all done!"

