#!/usr/bin/env bash
# should work find with bash, but ksh is better

#	Mnemonic:	run_op.ksh
#	Abstract:	This script will run all of the optimisation test problems
#				(four peaks, alternating bits, and graph colouring) capturing
#				the output in separate output files:  all_colour_n.out, all_4p.out
#				an all_altbit.out.  We assume that the necessary class and jar
#				files are in the ./class directory.  If running in a container, 
#				we assume that the directory with the needed stuff is in /data.
#
#				This script will attempt to build the .java source if the expected
#				class files are not present.  Java and javac are expected to be
#				installed.
#
#	Date:		20 February 2019
#	Author: 	Edward Scott Daniels  edaniels@gatech.edu
#
#	NOTE:		Running all tests takes between 4 and 5 hours:
#
#					/tmp/test $ mksh /data/src/a2/run_op.ksh
#					java Try_mine -F -t 1 -n 100 -r 2  
#					82m51.10s real    84m09.97s user     0m22.02s system
#					java Try_mine -C -t 1 -n 20 -r 10  
#					8m44.74s real     8m52.92s user     0m02.02s system
#					java Try_mine -C -t 1 -n 40 -r 20  
#					37m15.69s real    37m40.96s user     0m09.00s system
#					java Try_mine -C -t 1 -n 60 -r 30  
#					108m39.38s real   109m51.23s user     0m28.22s system
#					java Try_mine -A -t 1 -n 100 -r 2  
#					84m53.98s real    86m11.68s user     0m25.56s system
#
#
# -----------------------------------------------------------------------------


# run with $1==iterations and $2 the type (-h, -s...)
function run_it {
	if (( ! quiet )) 
	then
		echo "java Try_mine "$@" $train_data $test_data" >/dev/tty
	fi

	echo "# $(date)" 
	echo "# java Try_mine  "$@" $train_data $test_data"
	
	time java Try_mine  "$@" $train_data $test_data
}

# ---------------------------------------------------------------------------

#
#	ensure that the class files are there; if not build from my code.
#	
function validate_env {
	if [[ -d ./class ]]
	then
		class=$PWD/class
	else
		class=.
	fi

	if [[ ! -f $class/ABAGAIL.jar ]]
	then
		if [[ -z $CLASSPATH ]]
		then
			echo "[FAIL]  ABAGAIL must be present to build/run this code. ABAGAIL.jar is expected either in"
			echo "		this directory or in $PWD/class. Please find it and put it here."
		else
			echo "[WARN] ABAGAIL is not in $PWD/class; assuming it is in CLASSPATH"
		fi
		export CLASSPATH=.:$class:$CLASSPATH				# assume they have it in the exported path
	else
		export CLASSPATH=.:$class:$class/ABAGAIL.jar		# seems java needs . to complie?
	fi


	if [[ ! -e $class/Try_mine.class ]]		# my code is indeed Try_mine.java :)
	then
		mkdir -p ./class
		class=$PWD/class
		CLASSPAHT=$CLASSPATH:$class			# ensure it's there (hate java for this)
		(
			set -e	
			javac Try_mine.java >/tmp/javac.log 2>&1
			cp *.class $class/
		)
		if (( $? > 0 ))
		then
			echo "cannot find Try_mine.class, nor can it be built; giving up"
			exit 1
		fi

		echo "built successfully!"
	fi
}

# --------------------------------------------------------------------------------


out_dir=opt_probs_out
trials=1
while [[ $1 == -* ]]
do
	case $1 in
		-d) out_dir=$2; shift;;
		-t)	trials=$2; shift;;
	
		*)	echo "unrecognised parm: $1"
			echo "$0 [-d out-dir-path] [-t trials]"
			exit 1
			;; 
	esac
	shift
done

validate_env		# abort if our goodies aren't where they should be

mkdir -p $out_dir
echo "writing output files to out_dir"

# --- run all 4 algos on each propblem type -----------------------------
# four peaks
run_it -F -t $trials -n 100 -r 2 >$out_dir/all_4p.out

# Graph colouring (all three node settings (20, 40, 60)
run_it -C -t $trials -n 20 -r 10 >$out_dir/all_colour_20.out
run_it -C -t $trials -n 40 -r 20 >$out_dir/all_colour_40.out
run_it -C -t $trials -n 60 -r 30 >$out_dir/all_colour_60.out

# alternating bits
run_it -A -t 1 -n 100 -r 2 >$out_dir/all_altbits.out
