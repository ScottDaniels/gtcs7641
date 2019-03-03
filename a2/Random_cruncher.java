
//package opt.test;

import java.lang.*;
import java.io.*;
import java.util.*;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.net.URL;
import java.text.*;

/*
	Mnemonic:
	Abstract:	Implementation of randomised hill climbing, simulated annealing, and genetic
				algorithm specifially to look for optimal weights for assignment 1 ANN 
				research.   

				This code was reused (after being cleaned up) as 3rd generation code originally 
				written by Hannah Lau (source unknown) and adapted by Minming Zhao:
					https://github.com/minmingzhao/Adult_Salary_Classifier/tree/master/RandomOpimization

				Major modifications were made  to the code for:
					- readability, 
					- paramatarisation,
					- file loading effecency
					- clarity 

				The code was also augmented  to support a different dataset than the original autor
				used.

				Notes:
					From the Agagale doc, the options to the genetic class initialiser are:
						int populationSize,
						int toMate,
						int toMutate,
						GeneticAlgorithmProblem gap

	Enhancer:	Edward Scott Daniels
	Date:		15 February 2019
*/
 
public class Random_cruncher {
    private static Instance[] test_insts;			// data converted to abagale format
    private static Instance[] train_insts;
    
    private static int inputLayer = 10; 			// cooresponds to number of attributes
	private static int hiddenLayer = 5; 
	private static int outputLayer = 1; 			// one classification output
	private static int train_iter = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set;

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] names = { "RHC", "SA", "GA" };
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

	// ----------------------

	/*
		Count lines in a file so we don't have to know ahead of time...
		This, sadly, requires an additional pass, but it's easier than
		allocating and reallocating arrays later.
	*/
	private static int lif( String path ) {
		int lines = 0;
		BufferedReader br = null;

		try {
			br = new BufferedReader( new FileReader( path )); 
  			while( br.readLine() != null) {
				lines++;
			}
			br.close();
		} catch( Exception e ) {
			return -1;
        }

		return lines;
	}

	/*
		(A less buggered up way to load the data than the original code.)

		Given a file name, create an abagale instance from each row of 
		attributes (parameters) and the instance type (kind/class).
		The number of lines from the file are reported on standard output
		for verification.

		The number of parameters is assumed to be the number of tokens in the
		first record less 1.
	*/
	private static Instance[] load_data( String fname ) {
		Instance[] insts;
        BufferedReader br;
		int nrec;
		int i = 0;
		int iidx = 0;
		double attrs[];
		String tokens[];
		double inst_type;		// the instance type, or class of instance
		String rec;
		int nattrs = -1;

		nrec = lif( fname );								// get lines in file
		if( nrec < 1 ) {
			System.out.printf( "[FAIL] input file seems empty: %s\n", fname );
			System.exit( 1 );
		}
		System.out.printf( "# %s has %d lines\n", fname, nrec );

        insts = new Instance[nrec];							// rows of data instances in an abagale object

        try {
			br = new BufferedReader( new FileReader( fname ) );

			iidx = 0;
			while( (rec = br.readLine()) != null ) {
				tokens = rec.split( "," );					// assume comma separated
				if( nattrs < 0 ) {
					nattrs = tokens.length - 1;				// take number of parms based on first line
				}

				if( tokens.length > nattrs ) {				// tokens will also have instance type at end
					attrs = new double[nattrs];
					for( i = 0; i < nattrs; i++ ) {
						attrs[i] = Double.parseDouble( tokens[i] );
					}
					inst_type = Double.parseDouble( tokens[nattrs] );		// last one is the instance type

					insts[iidx] = new Instance( attrs );						// stuff in the attributes just parsed
					insts[iidx].setLabel( new Instance( inst_type ) );			// allong with the classifier
					iidx++;
				}
			}
        } catch(Exception e) {
			System.out.printf( "[FAIL] input file buggered: %s: %s\n", fname, e.toString() );
			System.exit( 1 );
            //e.printStackTrace();	// these are just bad form!
        }

		return insts;
		}

	/*
		This oddly expects to use the training instances in the class rather than
		being passed in a set to train on.  That might make sense if the class
		were being created, but it seems it's not? (More reasons to hate Java)
	*/
    private static void train( OptimizationAlgorithm oa, BackPropagationNetwork network, String name ) {
		double error = 0;
		int i;
		int j;

        //System.out.println("\nError results for " + name + "\n---------------------------");
		System.out.printf( "# training starts for: %s iter=%d\n", name, train_iter );

        for( i = 0; i < train_iter; i++ ) {
            oa.train();

            for( j = 0; j < train_insts.length; j++ ) {
                network.setInputValues( train_insts[j].getData() );
                network.run();

                Instance output = train_insts[j].getLabel(), example = new Instance( network.getOutputValues() );
                example.setLabel( new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value( output, example );
            }

            //why??? System.out.println( df.format(error) );
        }
    }

	/*
		Given a start timestamp, return the elapsed time beween that timestamp
		and now. Time returned is converted to seconds.
	*/
	private static double elapsed( double timestamp ) {
		double elapsed;

		elapsed = System.nanoTime() - timestamp;
		return elapsed / 1000000000.0;
	}

    public static void main(String[] args) {
		double timestamp;					// timestamps for measuring elapsed time
		double train_time; 				// time to actually train the network
		double val_time;				// time to validate results on train/test data
		double val_time_test;
		double train_correct = 0; 		// accuracy counts when run on training data
		double train_incorrect = 0; 
		double test_correct = 0; 		// accuracy counts when run on testing data
		double test_incorrect = 0;
        double predicted; 				// convenience variables 
		double actual;
		int i;
		int j;
		String train_path = "data/adult_training.csv";				// files
		String test_path = "data/adult_testing.csv";
		int		argc = 0;
		boolean run_all = true;			// default to running all of them
		boolean	run_sa = false;
		boolean run_ga = false;
		boolean run_hc = false;
 		
		while( argc < args.length && args[argc].charAt( 0 ) == '-' ) {
			switch( args[argc].charAt( 1 ) ) {
				case 'H':
					hiddenLayer = Integer.parseInt( args[argc+1] );
					argc++;
					break;

				case 'h':
					run_hc = true;
					run_all = false;
					break;

				case 'i':
					train_iter = Integer.parseInt( args[argc+1] );
					argc++;
					break;

				case 'g':
					run_ga = true;
					run_all = false;
					break;

				case 's':
					run_sa = true;
					run_all = false;
					break;

				case '?':
					System.out.printf( "usage: java -c <class-path> Random_cruncher [-h] [-g] [-s] [-H hidden-size] [-i iterations]\n" );
					System.exit( 0 );
					break;

				default:
					System.out.printf( "[WARN] ignoring unrecognised command line option: %s\n", args[argc] );
					break;
			}

			argc++;
		}

		if( argc >= args.length -1 ) {
			System.out.printf( "[FAIL] missing data file names for train and test data\n" );
			System.exit( 1 );
		}

		if( argc < args.length  ) {		// shouldn't need to test, but parinoia sets in
			train_path = args[argc];
			argc++;
		}
		if( argc < args.length ) {
			test_path = args[argc];
		}

		train_insts = load_data( train_path );
 		test_insts = load_data( test_path );
		set = new DataSet( train_insts );

        for( i = 0; i < oa.length; i++ ) { 				// create one ANN for each randomiser
            networks[i] = factory.createClassificationNetwork( new int[] {inputLayer, hiddenLayer, outputLayer} );
            nnop[i] = new NeuralNetworkOptimizationProblem( set, networks[i], measure );
        }

		if( run_hc || run_all ) {								// set up each selected opimiser
			System.out.printf( "# init Hill Climbing\n" );
        	oa[0] = new RandomizedHillClimbing( nnop[0] );
		}
		if( run_sa || run_all ) {
			System.out.printf( "# init Simulated Annealing\n" );
        	oa[1] = new SimulatedAnnealing( 1E11, .95, nnop[1] );
		}
		if( run_ga || run_all ) {
			System.out.printf( "# init Standard Genetic algorithm\n" );
        	oa[2] = new StandardGeneticAlgorithm( 2000, 1000, 100, nnop[2] );
		}

		System.out.printf( "# running optimisers\n" );
        for( i = 0; i < oa.length; i++) {					// run what was requested on the command line, or all
			if( oa[i] == null ) {
				continue;
			}

			timestamp = System.nanoTime(); 
            train( oa[i], networks[i], names[i] );			// fit to the model
            train_time = elapsed( timestamp );

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            timestamp = System.nanoTime();
            for( j = 0; j < train_insts.length; j++ ) {
                networks[i].setInputValues( train_insts[j].getData() );
                networks[i].run();													// run the network on the training data

                actual = Double.parseDouble( train_insts[j].getLabel().toString() );
                predicted = Double.parseDouble( networks[i].getOutputValues().toString() );	// is returned as a value 0.0 - 1.0
				predicted = predicted >= .5 ? 1.0 : 0.0;									// prediction is a real 0-1; assume > .5 to be 'true'
				if( actual == predicted ) {
					train_correct++;
				} else {
					train_incorrect++;
				}
				
            }
            val_time = elapsed( timestamp );
            
            timestamp = System.nanoTime();
            for( j = 0; j < test_insts.length; j++) {
                networks[i].setInputValues(test_insts[j].getData());
                networks[i].run();													// run the network on the test data

                actual = Double.parseDouble( test_insts[j].getLabel().toString() );
                predicted= Double.parseDouble( networks[i].getOutputValues().toString() );
				predicted = predicted >= .5 ? 1.0 : 0.0;									// prediction is a real 0-1; assume > .5 to be 'true'
				if( actual == predicted ) {
					test_correct++;
				} else {
					test_incorrect++;
				}
            }
            val_time_test = elapsed( timestamp );

			// this sucks, but I didn't have time (take the time) to change the original code
            results +=  "\nResults for " + names[i] + ": \nTraining: Correctly classified " + train_correct + " instances." +
                        "\nTraining: Incorrectly classified " + train_incorrect + " instances.\nTraining: Percent correctly classified: "
                        + df.format(train_correct/(train_correct+train_incorrect)*100) + "%\nTraining: Training time: " + df.format(train_time)
                        + " seconds\nTraining: Testing time: " + df.format(val_time) + " seconds\nTesting: Correctly classified " + 
                        test_correct + "instances.\nTesting: Incorrectly classified " + test_incorrect + "instances.\nTesting: Percent correctly classified: " + 
                        df.format(test_correct/(test_correct+test_incorrect)*100) + "%\nTest: Test time: " + df.format(val_time_test) + " seconds\n";

        }

        System.out.println(results);
    }

}

