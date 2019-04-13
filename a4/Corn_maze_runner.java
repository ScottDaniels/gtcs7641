
/*
	Mnemonic:	Corn_maze_runner
	Abstract:	This class implements verious experiment drivers
				that the main programme can start given the domain
				it builds.
	Author:		Edward Scott Daniels
	Date:		30 March 2019

	Acknowledgement:  This code is based on samples from:
					https://github.com/minmingzhao/Adult_Salary_Classifier
				but wildly extended to pull and compare values from the 
				various underlying policies generated.
*/

// I really HATE java's nickle and diming every bleeding class reference like this. 
// why can't we have a simple import <burlap>? I'm sure java programmers would have
// listed every single class rather than all the .*s, but I see no need for such 
// AR behavour.

import burlap.statehashing.*;
import burlap.statehashing.simple.*;
import burlap.behavior.policy.*;
import burlap.behavior.singleagent.planning.stochastic.*;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.core.state.State;
import burlap.behavior.singleagent.*;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.*;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.*;
import burlap.mdp.singleagent.SADomain;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.*;
import burlap.behavior.singleagent.auxiliary.*;
import burlap.domain.singleagent.gridworld.*;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.tdmethods.*;
import burlap.mdp.singleagent.environment.*;
import burlap.behavior.stochasticgames.madynamicprogramming.policies.*;
import burlap.behavior.singleagent.planning.deterministic.*;
import burlap.behavior.singleagent.planning.deterministic.informed.*;

import java.util.List;

public class Corn_maze_runner {
	public static final int UNKNOWN = 0;		// unknown will use egreedy but with warning
	public static final int EGMAX = 1;			// policy types for qlearning start
	public static final int GREEDYQ = 2;
	public static final int SDPLANNER = 3;
	public static final int EGREEDY = 4;		// sepcifically selected so no warning of default
	public static final int BOLTZ = 5;
	public static final int GREEDYD = 6;			// greedy deterministic q

	final SimpleHashableStateFactory hashers = new SimpleHashableStateFactory();

	Corn_maze univ;				// the cornmaze universe (map, term/reward stuff...)
	SADomain dom;				// gridword stuff

	/*
		The object is simple. We just take a domain and a 'universe' which is
		a Corn_maze (pretty much just the map).
	*/
	public Corn_maze_runner( Corn_maze cm, SADomain domain ) {
		univ = cm;
		dom = domain;
	}

	/*
		Start an x-windows based visualisaton which shows the universe and their values. The
		rendering colourises each cell based on value (red lowest, purple medium, blue
		higest). The policy (wich is the preferred direction from the cell) can also
		be overlaid.

		CAUTION:  this requires and X-like windowing, so containerised operation may
			be dodgy if impossible. Use get_values() and print_values() (below) to
			get and print to the tty (no pretty colours or policy action, but it's
			better than nothing!
	*/
	public void show_vfunct( ValueFunction vf, Policy p, State st, SADomain domain, 
							 HashableStateFactory hf, int[] dimensions ) {
		List<State> all_st;
		ValueFunctionVisualizerGUI gui;

		all_st = StateReachability.getReachableStates( st, domain, hf );
		gui = GridWorldDomain.getGridWorldValueFunctionVisualization( all_st, dimensions[0], dimensions[1], vf, p );
		gui.initGUI();
	}

	/*
		Crank out a set of values for the universe. This expects a value function which is
		many of the iteration objects (policy, value) which can return a value for any
		given state.   Here we just take the list of states, query their location (x,y)
		and then plug the state value as reported by the vf into the array. 

		We do need the list of states; even though we are passed a policy or value iteration
		object, which can produce states, we have to accept the common denominator object
		(value function) which does not have a get states function.
	*/
	private static double[][] get_values( ValueFunction vf, List<State> all_st, Policy p, State first_st, int[] dimensions ) {
		double[][] values;
		int x;
		int y;
		int i;
		State st;

		values = new double[dimensions[0]][dimensions[1]];

		for( i = 0; i < all_st.size(); i++ ) {
			st = all_st.get( i );
			x = Integer.valueOf( st.get( "x" ).toString() );	// could this be any more complex?
			y = Integer.valueOf( st.get( "y" ).toString() );
			values[x][y] = vf.value( st );
		}

		return values;
	}

	/*
		This prints the matrix of values passed in to the tty.
	*/
	public static void print_values( double[][] values ) {
		int x;
		int y;

		if( values == null ) {
			return;
		}

		for( y = values[0].length - 1; y >= 0; y-- ) {
			for( x = 0; x < values.length; x++ ) {
				System.err.printf( "%6.2f ", values[x][y] );
			}
			System.err.printf( "\n" );
		}
	}

	/*
		Given two value sets, return true if they match; false if not.
		(This can be made faster, but does for these simple tests.)
		We compare equal if within 0.05.
	*/
	public static boolean values_match( double[][] v1, double[][] v2 ) {
		int dim_x;
		int dim_y;
		int x;
		int y;

		if( v1.length <= 0 || v2.length <= 0 ) {
			return false;
		}

		if( v1[0].length <= 0 || v2[0].length <= 0 ) {
			return false;
		}

		if( v1.length != v2.length ) {
			return false;
		}

		if( v1[0].length != v2[0].length ) {
			return false;
		}

		dim_x = v1.length;
		dim_y = v1[0].length;

		//for( y = dim_y-1; y >= 0; y-- ) {
		for( y = 0; y < dim_y; y++ ) {
			for( x = 0; x < dim_x; x++ ) {
				if( Math.abs( v1[x][y] - v2[x][y] ) > .05  ) {
					return false;
				}
			}
		}

		return true;
	}

	/*
		Similar to value_match, this finds the cell in each matrix with
		the largest difference and returns that as the overall difference
		of the two. 

		Caution, an error returns -99999. 
	*/
	public static double values_diff( double[][] v1, double[][] v2 ) {
		int dim_x;
		int dim_y;
		int x;
		int y;
		double max_diff = 0.0;
		double v;

		if( v1.length <= 0 || v2.length <= 0 ) {
			return -99999;
		}

		if( v1[0].length <= 0 || v2[0].length <= 0 ) {
			return -99999;
		}

		if( v1.length != v2.length ) {
			return -99999;
		}

		if( v1[0].length != v2[0].length ) {
			return -99999;
		}

		dim_x = v1.length;
		dim_y = v1[0].length;

		//for( y = dim_y-1; y >= 0; y-- ) {
		for( y = 0; y < dim_y; y++ ) {
			for( x = 0; x < dim_x; x++ ) {
				if( (v = Math.abs( v1[x][y] - v2[x][y] )) > max_diff  ) {
					max_diff = v;
				}
			}
		}

		return max_diff;
	}

	/*
		Runs a value iteration analysis. Returns an array of the same dimensions
		of the universe with the values of each element.

		Gamma is expected to be 0<= g <1 and is NOT vetted here. 
		If vis is true, then we'll pop open a GUI and show the results; see caution 
		above. 
	*/
	public double[][] run_va( int[] iter_set, State start_state, double gamma, boolean vis ) {
		ValueIteration vi;
		EnumerablePolicy p = null;
		int[] dim;
		int iters;
		List<State> all_st;
		State st;					// must be our state as we need get_int()
		int x;
		int y;
		double[][] values;
		
		iters = iter_set[0];		// we only have one iteration value

		vi = new ValueIteration( dom, gamma, hashers, -1, iters );
		p = vi.planFromState( start_state );					// build the policy

		dim = univ.get_dimensions( );
		all_st = StateReachability.getReachableStates( start_state, dom, hashers );
		values = get_values( vi, all_st, p, start_state, dim );

		//System.err.printf( "value iter reports %d iterations\n", vi.getTotalValueIterations() );
		if( vis ) {
			show_vfunct( (ValueFunction) vi, p, start_state, dom, hashers, dim );
		}

		return values;
	}

	/*
		Runs a policy iteration analysis. The iter set is either two or three
			values.  We'll use the last two as:
				e_iters == evaluation iterations
				p_iters == policy iterations

		Gamma is expected to be 0<= g <1 and is NOT vetted here. 
		If vis is true, then we'll pop open a GUI and show the results; see caution 
		above. 
	*/
	public double[][] run_pa( int[] iter_set, State start_state, double gamma, boolean vis ) {
	 	PolicyIteration pi = null;		// mmmmm pi; blueberry
		EnumerablePolicy p = null;
		int[] dim;
		double max_pid_delta = -1.0;
		double max_eval_delta = 1.0;
		int e_iters = 100;
		int p_iters = 100;
		int i = 0;
		double[][] values;				// the values of the universe elements
		List<State> all_st;		// all known states

		if( iter_set.length > 1 ) {
			if( iter_set.length == 3 ) {
				i++;
			}
			e_iters = iter_set[i];
			if( i+1 >= iter_set.length ) {
				p_iters = iter_set[i+1];
			}
		}
		
		pi = new PolicyIteration( dom, gamma, hashers, max_pid_delta, max_eval_delta, e_iters, p_iters );
		p = pi.planFromState( start_state );					// build the policy

		// the iterator will use all given, so this is somewhat meaningless unless there is another
		// stopping point that can be given that I've not found.
		//System.err.printf( "policy iters: %d iterations\n", pi.getTotalPolicyIterations() );
		//System.err.printf( "value iters: reports %d iterations\n", pi.getTotalValueIterations() );
		dim = univ.get_dimensions( );
		all_st = StateReachability.getReachableStates( start_state, dom, hashers );
		values = get_values( pi, all_st, p, start_state, dim );

		if( vis ) {
			show_vfunct( (ValueFunction) pi, p, start_state, dom, hashers, dim );
		}

		return values;
	}
 
	/*
		Run a q learner on the universe (corn maze).

		Things available from episode:
			numActions()  # actions taken (1 less than num states visited)
			numTimeSteps()  # states visited
			reward(int t)   # reward received at time t

		Policy is one of the exploration method (e.g. EGMAX) constants.

		The iter set is used as follows:
			[0] is the number of iterations the learner is asked to run
			[1] is the number of invocations of the learner and thus the number of 
				solution paths evaluated; information on the best path found is 
				written to the standard error device.
	*/
	public void run_ql( int[] iter_set, State start_state, double gamma, boolean vis, int policy, double epsilon ) {
		QLearning agent = null;
		Policy p = null;
		Episode epi = null;						// a description of what the agent learned
		int iters;								// number of iterations to run
		long start_ts; 							// timestamp that an iteration started
		long elapsed;							// time required by an iteration 
		int i;									// see fortran :) 
		int j;
		int k;
		int x;									// map position
		int y;
		int[] dim;								// map dimensions
		double qinit = 0.0;
		double learn_rate = 1.0;
		SimulatedEnvironment env;				// environment which runs things
		double rtotal;
		Episode best_epi = null;
		State s;
		int best_i = -1;
		double best_r = -9000.0;
		BoltzmannQPolicy blp;
		SDPlannerPolicy sdp;
		GreedyQPolicy gq;
		GreedyDeterministicQPolicy gd;
		int rounds = 100;

		dim = univ.get_dimensions( );			// get dimensions for showing
		iters = iter_set[0];
		if( iter_set.length > 1 && iter_set[1] > 0 ) {
			rounds = iter_set[1];
		}

		env = new SimulatedEnvironment( dom, start_state );
		agent = new QLearning( dom, gamma, hashers, qinit, learn_rate );		// default qlearner with epsilon-greedy exploration
		
		switch( policy ) {		// set alternate exploration policy if indicated
			case EGMAX:
				agent.setLearningPolicy( new EGreedyMaxWellfare( epsilon ) );
				break;

			case GREEDYD:		// this sucked 3 of 4 CPUs and produced no results after several minutes, so abandoned
				System.err.printf( "# running random policy\n" );
				gd = new GreedyDeterministicQPolicy( );
				//gd.setSolver( (MDPSolverInterface ) new PolicyIteration( dom, gamma, hashers, -1.0, 1.0, 100, 100 ) );
				gd.setSolver( (MDPSolverInterface ) new ValueIteration( dom, gamma, hashers, -1.0,  iters ) );
				agent.setLearningPolicy( gd );
				break;

			case GREEDYQ:		// known to work
				System.err.printf( "# running greedy Q policy\n" );
				gq = new GreedyQPolicy( );
				gq.setSolver( (MDPSolverInterface ) new ValueIteration( dom, gamma, hashers, -1.0,  iters ) );
				agent.setLearningPolicy( gq );

				break;

			case BOLTZ: 			// known to work
				blp = new BoltzmannQPolicy( epsilon );
				blp.setSolver( (MDPSolverInterface ) new PolicyIteration( dom, gamma, hashers, -1.0, 1.0, iters, iters ) );
				agent.setLearningPolicy( blp );
				break;

			case UNKNOWN:
				System.err.printf( "using default exploration: epsilongreedy\n" );
				// fall through
			default:
				if( epsilon != 0.1 ) {		// set higher if not the same as the package default
					agent.setLearningPolicy( new EpsilonGreedy( agent, epsilon ) );
				}
				break;
		}

		for( i = 0; i< rounds; i++ ) {
			start_ts = System.nanoTime();
			epi = agent.runLearningEpisode( env );	// run one learning 'pass' and get an episode
			elapsed = System.nanoTime() - start_ts;

			rtotal = 0;
			for( j = 1; j < epi.numTimeSteps(); j++ ) {
				rtotal += epi.reward( j );					// sum the rewards received
			}

			System.err.printf( "%3d %6d (mu-s)  %3d %6.3f\n", i, elapsed/1000, epi.numTimeSteps(), rtotal );
			
			if( best_epi == null || epi.numTimeSteps() < best_epi.numTimeSteps()  ) {
				best_epi = epi;
				best_i = i;
			}
			if( rtotal > best_r ) {
				best_r = rtotal;
			}

			env.resetEnvironment();			// return simulation to initial state
		}

		epi = best_epi;
		System.err.printf( "Best iteration: %d    higest reward: %.3f\n", best_i, best_r );
		for( k = 0; k < epi.actionSequence.size(); k++ ) {
			s = epi.stateSequence.get( k );
			x = Integer.valueOf( s.get( "x" ).toString() );
			y = Integer.valueOf( s.get( "y" ).toString() );
		
			if( k % 10 == 0 ) {
				//System.err.printf( "\n%5s(%2d,%2d) ", epi.actionSequence.get( k ).actionName(), x, y );
				System.err.printf( "\nPATH: %d,%d ",  x, y );
			} else {
				//System.err.printf( "%5s(%2d,%2d) ", epi.actionSequence.get( k ).actionName(), x, y );
				System.err.printf( "%d,%d ",  x, y );
			}
		}
		System.err.printf( "\n" );

		if( vis ) {
			agent.initializeForPlanning( iters );
			p = agent.planFromState( start_state );
			show_vfunct( (ValueFunction) agent, p, start_state, dom, hashers, dim );
		}
	}
}
