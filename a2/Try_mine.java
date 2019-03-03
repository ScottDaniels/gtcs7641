
//package opt.test;

/*
	Mnemonic:	Try_mine.java
	Abstract:	This is my implementation to drive the four randomised optimisation
				algorithms which are being evaluated: (randomised hill climbing,
				simulated annealing, MIMIC and genetic algorithm).

	Author:		Edward Scott Daniels
	Date:		16 February 2019

	Notes:		The ranges array is passed seemingly everywhere in Abigale with
				little or no explaination about what it actually represents. 
				Looking at the code, it is the range for each 'attribute' 0 to
				n EXCLUDING n, and is the range that each element in the bit
				string may be randomised over. 

	Credit:		The base for this code comes from a github opensource project:
				https://github.com/minmingzhao/Adult_Salary_Classifier/blob/master/RandomOpimization/FourPeaksTest_MMZ.java
				Changes here were mostly to:
					- improve readability 
					- split into functions allowing for single type (hill climbing, sa, ...) execution
					- support all three problems being tested (Four peaks, k-colours. ???)
					- to add comments  (I abhore code that isn't commented espeically if someone else might use it)
					- allw for simple command line arguments to make expermentation easier
*/

//import java.text.DecimalFormat;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;

// requires Kcolour.class be in the class path, but no import is needed

public class Try_mine {
	private static final int KCOLOUR = 1;		// ef kinds
	private static final int FOURPEAKS = 2;
	private static final int ALTBITS = 3;

	private static final int RUN_ALL = 0;		// what it is we want to run; one specific or all
	private static final int RUN_MIMIC = 1;
	private static final int RUN_HILL = 2;
	private static final int RUN_SA = 3;
	private static final int RUN_GA = 4;

	private static boolean announced = false;	// for info we wish to write once to the tty
	

	// -------------- convenience functions ---------------------------------------------------------

	/*
		Given a start timestamp, return the elapsed time beween that timestamp
		and now. Time returned is converted to seconds.
	*/
	private static double elapsed( double timestamp ) {
		double elapsed;

		elapsed = System.nanoTime() - timestamp;
		return elapsed / 1000000000.0;
	}

	/*
		Given an evaluation function type (kind), return a new one.
	*/
	private static EvaluationFunction mk_ef( int kind, int nele ) {
		switch( kind ) {
			case ALTBITS:
				if( ! announced ) {
					System.out.printf( "Running alt-bits experiments on %d nodes\n", nele );
					announced = true;
				}
				return new Alt_bits( );

			case KCOLOUR:
				if( ! announced ) {
					System.out.printf( "Running k-colour experiments on %d nodes\n", nele );
					announced = true;
				}
				return new Kcolour( nele );

			case FOURPEAKS:
				if( ! announced ) {
					System.out.printf( "Running four peaks experiments with %d nodes\n", nele );
					announced = true;
				}
				return new FourPeaksEvaluationFunction( nele / 10 );
	
			default: 
				System.out.printf( "internal mishap: unknown eval function type; %d\n", kind );
				System.exit( 1 );
		}

		return null;
	}

	static void usage( ) {
		System.out.printf( "usage: java [-cp class_path] Try_mine [-t trials] [-F|-C] [-s|-g|-m|-h] [-n num_elements] [-r range-init]\n" );
		System.out.printf( "\tWhere\n\t\t-F|-C selects Four peaks or K-colours\n" );
		System.out.printf( "\t\t-s|-g|-m|-h  selects  a single algorythm (sim. anealing, genetic, MIMIC, or hill climbing)\n" );
		System.out.printf( "\t\t-n sets the number of nodes in the graph to colour\n" );
		System.exit( 1 );
	}
    
	// ------------------ algorithm specific functions --------------------------------------------
	/*
		Setup and run the random hill climbing optimiser for the given evaluation
		function.
	*/
	static void do_rhc( EvaluationFunction ef, int[] ranges, int trials  ) {
		int i;
		double timestamp;
        double train_time;
        RandomizedHillClimbing rhc;
        HillClimbingProblem hcp;
        FixedIterationTrainer fit;
		//ConvergenceTrainer fit;
        NeighborFunction nf;
        Distribution odd;
		int iterations[] = { 10, 35, 60, 100, 150, 250, 500, 1000, 50000, 100000, 200000, 500000 };

		while( trials > 0 ) {
			for( i = 0; i < iterations.length; i++ ) {
				odd = new DiscreteUniformDistribution( ranges );
				nf = new DiscreteChangeOneNeighbor( ranges );
				hcp = new GenericHillClimbingProblem( ef, odd, nf );
				rhc = new RandomizedHillClimbing( hcp );      
				fit = new FixedIterationTrainer( rhc, iterations[i] );	
        		//fit = new ConvergenceTrainer( rhc, 0, 200000);
		
        		timestamp = System.nanoTime();
        		fit.train();
				train_time = elapsed( timestamp );
		
        		//double fit_its = fit.getIterations();
        		System.out.printf("rhc:  time: %7.3f   max: %7.3f iter=%d\n", train_time, ef.value( rhc.getOptimal() ), iterations[i] );
        		//System.out.printf("rhc:  time: %7.3f   max: %7.3f iter=%.2f\n", train_time, ef.value( rhc.getOptimal() ), fit_its );
			}
			
			trials--;
		}
	}
        
	/*
		Setup and run a simulated annealing optimiser for the given evaluation function and set
		of ranges.

	*/
	static void do_sa( EvaluationFunction ef, int[] ranges, int trials ) {
		double timestamp;
        double train_time;
		int i;
		int j;
       	SimulatedAnnealing sa;
        HillClimbingProblem hcp;
        Distribution odd;
        NeighborFunction nf;
        FixedIterationTrainer fit;
        double[] cooling_size = { 0.25, 0.35, 0.5, 0.75, 0.80, 0.99 };			// sa cooling amount; tweaked for each iteration
		int iterations[] = {   10, 35, 60, 100, 250, 500, 1000, 5000, 100000, 200000, 500000 };

		while( trials > 0 ) {
			nf = new DiscreteChangeOneNeighbor( ranges );
			odd = new DiscreteUniformDistribution( ranges );
			hcp = new GenericHillClimbingProblem( ef, odd, nf );
	
        	for( i = 0; i < cooling_size.length; i++) {
				for( j = 0; j < iterations.length; j++ ) {
					sa = new SimulatedAnnealing( 1E11, cooling_size[i], hcp );
        			fit = new FixedIterationTrainer( sa, iterations[j] );
        			//fit = new ConvergenceTrainer(sa,0,1000);
        			timestamp = System.nanoTime();
        			fit.train();
        			train_time = elapsed( timestamp );
        			//fit_its = fit.getIterations();
		
        			System.out.printf("sa:   time: %7.3f   max: %7.3f cooling: %.3f t=%d iter=%d\n", 
						train_time, ef.value( sa.getOptimal() ), cooling_size[i], trials, iterations[j] );
				}	
        	}
		
			trials--;
		}
        
	}

	static void do_genetic( EvaluationFunction ef, int[] ranges ) {
		double timestamp;
        double train_time;
		int i;
		int j;
		int k;
		int l;
		int n;
        FixedIterationTrainer fit;
        //ConvergenceTrainer fit;
        StandardGeneticAlgorithm ga;
        GeneticAlgorithmProblem gap;
        MutationFunction mf;
        CrossoverFunction cf;
        Distribution df;
        Distribution odd;
		//GenericProbabilisticOptimizationProblem pop;		// these class names are just horrible!
        ProbabilisticOptimizationProblem pop;
        double[] pop_size = { 20, 40, 60, 100, 120, 140, 200 };

        //double[] to_mate_size = { 0.1, 0.2, 0.5, 0.8, 1 };
        double[] to_mate_size = { 0.1, 0.2, 0.5 };

        //double[] to_mutation_size = { 0.001, 0.05, 0.2, 0.5, 0.8, 1 };
        double[] to_mutation_size = { 0.001, 0.05, 0.2 };
		int iterations[] = { 10, 35, 60, 100, 250, 500, 1000, 1500, 2000, 3000, 5000 };

		n = ranges.length;

		for( i = 0; i < pop_size.length; i++) {
			for( j = 0; j < to_mate_size.length; j++) {
				for( k = 0; k < to_mutation_size.length; k++) {
					for( l = 0; l < iterations.length; l++ ) {
						mf = new DiscreteChangeOneMutation( ranges );		// it seems we should have a new one of each for each go
						cf = new SingleCrossOver();							// but nothing in the doc gives indication one way or the other
						df = new DiscreteDependencyTree(.1,  ranges ); 
						odd = new DiscreteUniformDistribution( ranges );
				
						gap = new GenericGeneticAlgorithmProblem( ef, odd, mf, cf );
						pop = new GenericProbabilisticOptimizationProblem( ef, odd, df );

						ga = new StandardGeneticAlgorithm((int) ( n * pop_size[i]), 
									(int) (n * pop_size[i] * to_mate_size[j] ), 
									(int) (n * pop_size[i] * to_mutation_size[k]), gap );
						fit = new FixedIterationTrainer( ga, iterations[l] );
						//fit = new ConvergenceTrainer( ga, 0, 1000);

						timestamp = System.nanoTime();
						fit.train();
						train_time = elapsed( timestamp );

						//fit_its = fit.getIterations();
						System.out.printf( "ga:   time: %7.3f   max: %7.3f pop: %.3f tom_size: %.3f mut_size: %.3f iter=%d\n", 
							train_time, ef.value( ga.getOptimal() ), pop_size[i], to_mate_size[j], to_mutation_size[k], iterations[l] );
					}
				}
			}
		}
	}

	/*
		Setup and run a mimic optimiser for the given evaluation function and
		set of ranges.
	*/
	static void do_mimic( EvaluationFunction ef, int[] ranges ) {
		int i;
		int j;
		int k;
		MIMIC mimic;
		double timestamp;
        double train_time;
        ProbabilisticOptimizationProblem pop;
        Distribution odd;
        Distribution df;
        double[] pop_size = { 20, 40, 60, 80, 100, 120, 140, 200 };
        //double[] pop_size = {  200, 400, 600 };
        double[] pop_size_kept = { 0.05, 0.075, 0.1, 0.2, 0.5 };
        //double[] pop_size_kept = { 0.05, 0.075, 0.1, 0.2, 0.5, 0.7, 0.9 };	// this gets a little too long to run 
		int iterations[] = { 10, 35, 60, 100, 250, 500, 1000, 1500, 2000, 3000, 5000 };

        FixedIterationTrainer fit;
        //ConvergenceTrainer fit;
		int req_iters = 0;				// iterations requred to converge


		for( i = 0; i < pop_size.length; i++) {
			for( j = 0; j < pop_size_kept.length; j++) {
				for( k = 0; k < iterations.length; k++ ) {
					df = new DiscreteDependencyTree( 0.1, ranges ); 
					odd = new DiscreteUniformDistribution( ranges );
					pop = new GenericProbabilisticOptimizationProblem( ef, odd, df );

					mimic = new MIMIC( (int) (pop_size[i]), (int) (pop_size[i] * pop_size_kept[j]), pop );
					fit = new FixedIterationTrainer( mimic, iterations[k] );
					//fit = new ConvergenceTrainer( mimic, 0, 10000);

					timestamp = System.nanoTime();
					fit.train();
					train_time = elapsed( timestamp );
					//req_iters = fit.getIterations();
					System.out.printf( "mimic: time: %7.3f   max: %7.3f pop-size: %.3f %%kept: %.3f iter=%d\n", 
							train_time, ef.value(mimic.getOptimal() ), pop_size[i], pop_size_kept[j], iterations[k] );
				}
			}

		}
	}

    public static void main(String[] args) {
		int kind = FOURPEAKS;		// default to four peaks, and set nele/rval accordingly as defaults too
    	int nele = 100;				// number of elements in the value array
    	int t = nele / 5;			// t value given to the peaks class (-n will adjust if given)
		int rinit = 2;				// default is for four peaks
		int argc;
        int[] ranges;				// this array specifies a "row" of state. In general, initialised to the range 0 to n-1.
        EvaluationFunction ef;		// the evaluation function to run
		int what2run = RUN_ALL;		// what algos to run (hill, sa, ...)
		int trials=1;				// number of trials (-t) for some types of algos
	

		argc = 0;
		while( argc < args.length && args[argc].charAt( 0 ) == '-' ) {		// args give some control on command line
			switch( args[argc].charAt( 1 ) ) {
				case '?':
					usage();
					System.exit( 0 );
					break;

				case 'A':
					kind = ALTBITS;
					break;

				case 'C':
					kind = KCOLOUR;
					break;

				case 'F':
					kind = FOURPEAKS;
					break;

				case 'h':
					what2run = RUN_HILL;
					break;

				case 'g':
					what2run = RUN_GA;
					break;

				case 'm':
					what2run = RUN_MIMIC;
					break;

				case 'n':			// pick up n and set t (blindly assume value given; they'll get a dump if not; their loss
					nele = Integer.parseInt( args[argc+1] );
					t = nele  /5;
					argc++;
					break;

				case 'r':			// initialisation value for range
					rinit = Integer.parseInt( args[argc+1] );
					argc++;
					break;

				case 's':
					what2run = RUN_SA;
					break;

				case 't':
					trials = Integer.parseInt( args[argc+1] );
					argc++;
					break;

				default:
					System.out.printf( "unrecognised command line option; %s\n", args[argc] );
					usage( );
					break;
			}

			argc++;
		}

		ranges = new int[nele];
        Arrays.fill( ranges, rinit );

		if( what2run == RUN_ALL || what2run == RUN_HILL ) {
			ef = mk_ef( kind, nele );								// select and initialise the eval function
			System.out.printf( "starting random hill climbing\n" );
			do_rhc( ef, ranges, trials  );
		}

		if( what2run == RUN_ALL || what2run == RUN_SA ) {
			ef = mk_ef( kind, nele );
			System.out.printf( "starting simulated annealing\n" );
			do_sa( ef, ranges, trials );
		}

		if( what2run == RUN_ALL || what2run == RUN_GA ) {
			ef = mk_ef( kind, nele );
			System.out.printf( "starting standard genetic\n" );
			do_genetic( ef, ranges );
		}

		if( what2run == RUN_ALL || what2run == RUN_MIMIC ) {
			ef = mk_ef( kind, nele );
			System.out.printf( "starting mimic rint=%d\n", rinit );
			do_mimic( ef, ranges );
		}
	}
}
