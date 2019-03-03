#!/usr/bin/env bash
# prefer kshell, but this is bash-able

#	Mnemonic:	run_cruncher.ksh
#	Abstract:	Run the "ranom cruncher" which drives any or all of the 
#				random optimisation algorithms as the weight setter for
#				the MLP classification ANN across the adult dataset. Once
#				weights are determined, the ANN is run to generate an
#				accuracy result.
#
#
#				NOTE:  the genetic algorithm portion alone takes about 
#					5 hours to run. The others are shorter, but if you
#					run them all, be prepared to wait.
#
#	Date:		16 February 2019
#	Author:		Edward Scott Daniels  edaniels7@gatech.edu
#	
# -----------------------------------------------------------------------

# run with $1==iterations and $2 the type (-h, -s...)
function run_it {
	if (( ! quiet )) 
	then
		echo "java Random_cruncher -i $1 $2 $train_data $test_data" >/dev/tty
	fi

	echo "# $(date)" 
	echo "# java Random_cruncher -i $1 $2 $train_data $test_data"
	
	time java Random_cruncher -i $1 $2 $train_data $test_data
}

function validate_data {
	if [[ ! -f $train_data ]]
	then
		echo "[FAIL] cannot find the training data set: $train_data"
		exit 1
	fi

	if [[ ! -f $test_data ]]
	then
		echo "[FAIL] cannot find the validation data set: $test_data"
		exit 1
	fi
}


function validate_env {
set -x
	if [[ -d class ]]
	then
		class=$PWD/class
	fi

	if [[ -d data ]]
	then
		data_dir=.				# assume downloaded directory
	else
		data_dir=/data			# assume running in container my container
	fi

	if [[ -d ./class ]]			# assume class directory has what we need
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

	if [[ ! -e $class/Random_cruncher.class ]]		# my code is indeed Random_cruncher.java :)
	then
		mkdir -p class
		class=$PWD/class

		(
			set -e	
			javac Random_cruncher.java >/tmp/javac.log 2>&1
			cp *.class $class/
		)
		if (( $? > 0 ))
		then
			echo "cannot find Random_cruncher.class, nor can it be built; giving up"
			exit 1
		fi

		echo "built successfully!"
	fi
set +x
}

# -----------------------------------------------------------------------

out_dir=cruncher_out
all=1
quiet=0
while [[ $1 == -* ]]
do
	case $1 in 
		-H)	all=0; hill_only=1;;
		-S) all=0; sa_only=1;;
		-G)	all=0; ga_only=1;;

		-d)	out_dir=$2; shift;;
		-q) quiet=1;;

		*)	echo "unknown option: $1"
			echo "usage: $0 [-H|-G|-S] [-d out-dir] [-q]"
			echo "-H, -G ,-S run only the one algorithm to generate weights (hill, sim. anna., genetic)"
			echo "-q supresses additional chatter to the tty so it can be run detached"
			exit 1
	esac

	shift
done

validate_env			# ensure our goodies are where we expect them; abort if not

train_data=$data_dir/data/adult_train.csv
test_data=$data_dir/data/adult_test.csv
validate_data			# will abort if one of the datasets is missing


if ! mkdir -p $out_dir
then
	echo "[FAIL] unable to create output directory: $outdir"
	exit 1
fi


if (( all || hill_only ))
then
	# run hill climbing and sa for a range of iterations
	for i in 10 100 250 500 1000 10000
	do
		run_it $i -h
	done >$out_dir/rc_hill.out
fi

if (( all || sa_only ))
then
	for i in 10 100 250 500 1000 10000 
	do
		run_it $i -s
	done >$out_dir/rc_sa.out
fi

# warning -- this takes about 5 hours to run:
#		10 iters == 5 min, 100 iters == 44 min, 200 iters == 88 min... 1000 iters == 440 min
if (( all || ga_only ))
then
	for i in 10 100 200 250 500 1000
	do
		run_it $i -g
	done >$out_dir/rc_ga.out
fi

touch done
date
