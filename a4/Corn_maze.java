
/*
	Mnemonic:	Corn_maze.java
	Abstract:	A gridworld implementation which supports several different
				map layouts (selected by command line parameter).
				(Here in Farmbille, rustic mid-western US) an annual fall
				tradition is to cut mazes into the corn stalks and let people
				attempt to navigate their way through them. Seems appropriate
				to name this experimental world Corn Maze. 

				Output from this code is to stderr because the burlap 
				library (annoingly) writes it's logging messages to stdout
				rather than stderr the way a library should. 

	Author:		Edward Scott Daniels   edaniels7
				cs7641 Spring 2019
	Date:		28 March 2019

	This code is based on the tutorials from the main burlap site:
		http://burlap.cs.brown.edu/tutorials/index.html
*/

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.StateTransitionProb;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.UniversalActionType;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.model.statemodel.FullStateModel;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.StatePainter;
import burlap.visualizer.StateRenderLayer;
import burlap.visualizer.Visualizer;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;



// ---------------------------------------------------------------------------
public class Corn_maze implements DomainGenerator {

	public static final String VAR_X = "x";
	public static final String VAR_Y = "y";

	public static final String ACTION_NORTH = "north";	// action strings
	public static final String ACTION_SOUTH = "south";
	public static final String ACTION_EAST = "east";
	public static final String ACTION_WEST = "west";
	
	public static final int NORTH = 0;			// directions
	public static final int SOUTH = 1;
	public static final int EAST = 2;
	public static final int WEST = 3;

	public static final int CLEAR_CELL = 0;
	public static final int WALL_CELL = 1;
	public static final int COSTLY_CELL = 2;
	public static final int NEG_EXTRACT_CELL = 3;
	public static final int GOAL_CELL = 9;

	public static final int EASY_MAP = 0;
	public static final int MED_MAP = 1;
	public static final int HARD_MAP = 2;
	public static final int FACTORY_MAP = 3;
	public static final int CITY_MAP = 4;
		

	protected RewardFunction rf;
	protected TerminalFunction tf;
	protected int[][] active_map;		// the selected map in use

	// ------------ possible maps that can be selected ----------------
	// oddly, the display funcitons rotate these 90 degrees anti-clockwise ????
	//ordered so first dimension is x
	protected int [][] default_map = new int[][]{
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0},
			{0,0,0,0,0,1,0,0,0,0,0},
			{1,0,1,1,1,1,1,0,0,1,1},
			{0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,0,0,0,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
			{0,0,0,0,1,0,0,0,0,0,0},
	};

	// few states, one barrier
	protected int [][] easy_map = new int[][]{
			{ 0,0,0,0,0,0 },
			{ 0,0,1,1,0,0 },
			{ 0,0,1,1,0,0 },
			{ 0,0,0,0,0,9 }
	};

	// large number of states, a few barriers
	protected int [][] medium_map = new int[][]{
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,1,1,1,1,1,1,0,0,1,1,1,1,1 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 1,1,1,1,1,0,0,1,1,1,1,1,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 1,0,0,1,1,1,0,0,0,1,1,1,1,1 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,1,1,1,1,1,1,1,1,1,1,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,9 }
	};

	// large number of states, a more barriers and multiple paths to goal from 0,0
	protected int [][] hard_map2 = new int[][]{
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,1,1,0,1,1,0,0,0,1,1,0,1,1 },
			{ 0,0,0,1,0,0,0,0,0,0,1,0,0,0 },
			{ 1,0,1,0,1,0,0,1,1,0,1,1,0,0 },
			{ 0,0,0,1,0,0,0,0,0,0,0,1,0,0 },
			{ 0,0,0,1,1,0,0,0,1,0,0,1,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 1,1,0,1,1,1,0,0,0,1,0,0,1,1 },
			{ 0,0,0,0,1,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,1,1,1,0,0,0,1,0,1,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,1,1,0,0,0,1,1,0,0,1,1,0,0 },
			{ 0,0,0,0,0,0,1,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,9,0 }
	};

	// large number of states, a more barriers and multiple paths to goal from 0,0
	// there is also a small path that is marked '2' which causes it to have an 
	// even more negitive reward to discourage use.
	protected int [][] hard_map = new int[][]{
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 0,1,1,1,1,1,1,0,0,1,1,1,1,1 },
			{ 0,0,0,1,0,0,0,0,0,0,1,0,0,0 },
			{ 1,0,1,1,1,0,0,1,1,1,1,1,0,0 },
			{ 0,0,0,1,0,0,0,0,0,0,0,1,0,0 },
			{ 0,0,0,1,1,1,1,1,1,0,0,1,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
			{ 1,1,0,1,1,1,0,0,0,1,1,1,1,1 },
			{ 0,0,0,0,1,0,0,0,0,0,0,0,0,0 },
			{ 0,0,0,0,1,1,1,1,1,1,1,1,1,0 },
			{ 0,0,0,0,2,2,2,2,2,2,0,0,0,0 },
			{ 0,1,1,1,1,1,1,1,1,1,1,1,0,0 },
			{ 0,0,0,0,0,0,1,0,0,0,0,0,0,0 },
			{ 0,0,0,0,0,0,0,0,0,0,0,0,9,0 }
	};

	// hard map used for write up. Almost 1000 states.
	//
	protected int [][] factory_map = new int[][] {
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0 },
	{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0 },
	{0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{1,1,1,1,1,1,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0 },
	{1,1,1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0 },
	{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0 },
	{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0 },
	{0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0 },
	{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0 },
	{1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
	{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9 }
	};

	// Easy map used for write up. About 200 states
	//
	protected int [][] city_map = {	
 	{ 1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,2,1,1,1,1,1,1,0,1,1,9,1,1 },
 	{ 1,1,1,0,0,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2 },
 	{ 1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,2 },
 	{ 1,1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,2 },
 	{ 1,1,1,0,1,1,0,0,1,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,2 },
 	{ 1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,2 },
 	{ 1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 2,2,2,0,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,2,2,2 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,2,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,2,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,2,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,2,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 1,1,1,0,1,1,1,1,0,1,1,2,1,1,1,1,1,2,1,1,1,1,1,1,0,0,0,0,0,0 },
 	{ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 0,1,1,0,1,1,1,1,1,1,1,2,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1 },
 	{ 0,1,1,0,1,1,1,1,1,1,1,2,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1 },
 	{ 0,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,0,1,1,1,1,1,2,2,2,2,2,2 }
	};


	// ---- tools ----------------------------------------------------------------------
	/*
		Convert without a stackdump if the value is not a digit.
	*/
	private static int atoi( String s ) {
		int v;

		try {
			v = Integer.parseInt( s );			
		} catch( Exception e ) {
			v = 0;					// we should do this right and parse up to non-num char, but no time
		}

		return v;
	}

	/*
		Parse it out and don't barf if user buggered it up. 
	*/
	private static double atof( String s ) {
		double v;

		try {
			v = Double.parseDouble( s );			
		} catch( Exception e ) {
			v = 0;					// we should do this right and parse up to non-num char, but no time
		}

		return v;
	}

	/*
		Expects a set of v,v,v... which are defrocked and placed into an array.
	*/
	private static int[] get_vset( String s, int def_val ) {
		int i;
		String[] tokens;
		int[] vset;
		int v;

		tokens = s.split( "," );

		vset = new int[tokens.length];
		for( i = 0; i < tokens.length; i++ ) {
			v = atoi( tokens[i] );
			if( v > 0 ) {
				vset[i] = v;
			} else {
				vset[i] = def_val;
			}
		}

		return vset;
	}

	// Put out x,y pairs of each square in the map that is a wall. Makes plotting 
	// with an external tool easier.  Also spits out the dimensions and number of
	// states in the map.
	//
	private void dump_walls( ) {
		int x; 
		int y;
		int [][] map;
		int	states = 0;

		map = active_map;

		for( x = 0; x < map.length; x ++ ) {
			System.err.printf( "WALLS: " );
			for( y = 0; y < map[0].length; y++ ) {
				if( map[x][y] >= 0 ) {
					System.err.printf( "%d,%d,%d ", x, y, map[x][y] );
				} else {
					states++;
				}
			}
			System.err.printf( "\n" );
		}

		System.err.printf( "\nSTATES: %d\n", states );
		System.err.printf( "\nDIM: %d %d\n", map.length, map[0].length );
	}

	// ---------------------------------------------------------------------------------
	public void select_map( int desired_map, boolean dump ) {
		switch( desired_map ) {
			case MED_MAP:
				this.active_map = medium_map;
				break;

			case HARD_MAP:
				this.active_map = hard_map2;
				break;

			case CITY_MAP:
				this.active_map = city_map;
				break;

			case FACTORY_MAP:
				this.active_map = factory_map;
				break;

			default:
				this.active_map = easy_map;
				break;
		}
	
		if( dump ) {
			dump_walls();
		}

		//System.err.printf( ">>>>> %d %d\n", this.active_map.length, this.active_map[0].length );
	}

	/*
		Returns the dimensions of the map as a two element array.
	*/
	public int[] get_dimensions( ) {
		int[] dim;

		dim = new int[2];

		if( active_map != null && active_map.length > 0 ) {
			dim[0] = active_map.length;
			dim[1] = active_map[0].length;
		} else {
			dim[0] = dim[1] = 0;
		}

		return dim;
	}

	/*
		Basic domain setup including establishing the reward and terminal funcitons.
		We set our reward/terminal objects into the model such that the model runner
		ends up calling our evaluation stuff.
	*/
	@Override
	public SADomain generateDomain() {
		SADomain domain;
		GridWorldStateModel smodel;

		domain = new SADomain();
		domain.addActionTypes(
				new UniversalActionType( ACTION_NORTH ),
				new UniversalActionType( ACTION_SOUTH ),
				new UniversalActionType( ACTION_EAST) ,
				new UniversalActionType( ACTION_WEST ) );


		smodel = new GridWorldStateModel();
		rf = new Reward_manager( this.active_map );
		tf = new Terminator( this.active_map );

		domain.setModel( new FactoredModel( smodel, rf, tf ) );		// push our stuff into the model

		return domain;
	}

	/*
		Allow the functions to be pulled out when needed.
	*/
	public TerminalFunction get_tf( ) {
		return tf;
	}
	public RewardFunction get_rf( ) {
		return rf;
	}

	public StateRenderLayer getStateRenderLayer(){
		StateRenderLayer rl = new StateRenderLayer();
		rl.addStatePainter(new Corn_maze.WallPainter());
		rl.addStatePainter(new Corn_maze.AgentPainter());


		return rl;
	}

	public Visualizer getVisualizer(){
		return new Visualizer(this.getStateRenderLayer());
	}

	// ---------------------------------------------------------------------------
	protected class GridWorldStateModel implements FullStateModel{
		protected double [][] transitionProbs;

		public GridWorldStateModel() {
			this.transitionProbs = new double[4][4];
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
				 double p = i != j ? 0.2/3 : 0.8;		// 80% on the diagnal, 6.6% all other places
				 transitionProbs[i][j] = p;
				}
			}
		}

		/*
			Given a state and action generate a list of possible state-prime objects
			which would result from the action.
		*/
		@Override
		public List<StateTransitionProb> stateTransitions( State s, Action a ) {
			int i;
			Corn_maze_state gs;
			int curX;			// location coord
			int curY;
			int adir;			// action direction
			List<StateTransitionProb> tps;
			StateTransitionProb noChange;
			int [] newPos;
			Corn_maze_state ns;		// new grid state

			gs = (Corn_maze_state) s; 			//get current location
			curX = gs.x;
			curY = gs.y;

			adir = actionDir( a );

			tps = new ArrayList<StateTransitionProb>( 4 );
			noChange = null;

			for( i = 0; i < 4; i++ ) {
				newPos = this.moveResult( curX, curY, i );

				if( newPos[0] != curX || newPos[1] != curY){
					ns = gs.copy(); 					// possible outcome
			 		ns.x = newPos[0];
			 		ns.y = newPos[1];

			 		//create transition probability object and add to our list of outcomes
			 		tps.add( new StateTransitionProb( ns, this.transitionProbs[adir][i] ) );
				} else{
			 		if( noChange != null) {				// agg if other directions also resulted in no movement
			  			noChange.p += this.transitionProbs[adir][i];
			 		} else{
			  			noChange = new StateTransitionProb( s.copy(), this.transitionProbs[adir][i] );
			  			tps.add( noChange );
			 		}
				}
			}

			return tps;
		}

		@Override
		public State sample( State s, Action a ) {
			Corn_maze_state gs;
			int curX;			// current position
			int curY;
			int adir;			// action direction
			double r;			// rand value for prob
			double sumProb;
			int [] newPos;		// position after move
			int dir;
			int i;

			s = s.copy();
			gs = (Corn_maze_state) s;
			curX = gs.x;
			curY = gs.y;

			adir = actionDir(a);	// translate dir to usable value

			r = Math.random(); 		//sample direction with random roll
			sumProb = 0.;
			dir = 0;
			for( i = 0; i < 4; i++ ){
				sumProb += this.transitionProbs[adir][i];
				if( r < sumProb ){
			 		dir = i;
			 		break;
				}
			}

			newPos = this.moveResult(curX, curY, dir);			// get resulting position
			gs.x = newPos[0];									//set the new position
			gs.y = newPos[1];

			return gs;					// send back modified state
		}

		/*
			Given an action, return an integer equiv.
		*/
		protected int actionDir( Action a ) {
			int adir = -1;
			String aname;

			aname = a.actionName();

			// future -- set this into a hash based on name (diagnal and 3d words add too many posible meoments
			if( aname.equals( ACTION_NORTH) ){
				adir = NORTH;
			} else {
				if( aname.equals( ACTION_SOUTH) ){
					adir = SOUTH;
				} else {
					 if( aname.equals( ACTION_EAST) ){
						adir = EAST;
					} else  {
						if( aname.equals( ACTION_WEST) ){
							adir = WEST;
						}
					}
				}
			}

			return adir;
		}


		/*
			Given a current x,y location, and a direction to move, compute a 
			new x,y based on the boundaries of the universe (map) and the 
			walls/barriers inside.  A new pair is returned.
		*/
		protected int[] moveResult( int cur_x, int cur_y, int direction ){
			int [][] map;
			int	nx = 0;					// new values
			int ny = 0;

			map = active_map;		// easy reference to the current map

		
			nx = cur_x;				// start at current pos
			ny = cur_y;

			switch( direction ) {			// adjust to get new x,y
				case NORTH:
					ny++;
					break;
	
				case SOUTH:
					ny--;
					break;

				case EAST:
					nx++;
					break;

				case WEST:
					nx--;
					break;
			}

			int width = map.length;
			int height = map[0].length;

			if( nx < 0 || nx >= width || ny < 0 || ny >= height || map[nx][ny] == 1){   // stay inside, off walls
				nx = cur_x;						// put back if invalid location
				ny = cur_y;
			}

			return new int[] { nx,ny };
		}
	}

	// ---------------------------------------------------------------------------
	public class WallPainter implements StatePainter {

		/*
			Paint each cell in the map.  Cells are coloured such that:
				black == wall/barrier
				yellow == higher cost 
				red == undesireable (negative) extraction
				green == desireable extraction (goal)
				white == clear path
		*/
		public void paint(Graphics2D g2, State s, float w_width, float w_height) {
			float fWidth;
			float fHeight;
			float rx;
			float ry;
			int [][] map;
			boolean is_black = false;

			map = active_map;

			fWidth = map.length; 						//set up floats for the width and height of our domain
			fHeight = map[0].length;

			float width = w_width / fWidth;				// scale to fit window size
			float height = w_height / fHeight;

			for(int i = 0; i < map.length; i++){				// look for walls and paint them
				for(int j = 0; j < map[0].length; j++){

					rx = i*width; 							//left coordinate of cell on our canvas
					ry = w_height - height - j*height;		// ajust for top right 0,0 origin

					switch( map[i][j] ) {
						case WALL_CELL:
							if( ! is_black ) {						// need to force colour
								g2.setColor(Color.BLACK);
							}
							g2.fill(new Rectangle2D.Float(rx, ry, width, height));
							break;

						case COSTLY_CELL:
							g2.setColor( Color.YELLOW );
							g2.fill(new Rectangle2D.Float(rx, ry, width, height));
							is_black = false;
							break;

						case NEG_EXTRACT_CELL:
							g2.setColor( Color.RED );
							g2.fill( new Rectangle2D.Float( rx, ry, width, height ) );
							is_black = false;
							break;

						case GOAL_CELL:
							g2.setColor( Color.GREEN );
							g2.fill( new Rectangle2D.Float( rx, ry, width, height ) );
							is_black = false;
							break;
					}
				}
			}
		}
	}


	// ---------------------------------------------------------------------------
	public class AgentPainter implements StatePainter {
		/*
			Paint the state. w_width, w_height describe the real dimensions into
			which we must scale.
		*/
		@Override
		public void paint(Graphics2D g2, State s, float w_width, float w_height) {
			int [][] map;
			float u_width;			// map (universe) dimensions
			float u_height;
			float c_width;			// cell dimensions
			float c_height;

			map = active_map;

			g2.setColor(Color.GRAY); 					//agent will be filled in gray

			u_width = map.length;
			u_height = map[0].length;

			c_width = w_width / u_width;					// scale cell to fit window
			c_height = w_height / u_height;

			int ax = (Integer)s.get(VAR_X);
			int ay = (Integer)s.get(VAR_Y);

			//left coordinate of cell on our canvas
			float rx = ax * c_width;

			//top coordinate of cell on our canvas
			//coordinate system adjustment because the java canvas
			//origin is in the top left instead of the bottom right
			float ry = w_height - c_height - (ay * c_height);

			//paint the rectangle
			g2.fill(new Ellipse2D.Float(rx, ry, c_width, c_height));
			//System.out.printf( ">>>> %d,%d\n", ax, ay );
		}
	}


	// ---------------------------------------------------------------------------
	/*
		A reward manager has a copy of the map, and given a state provides
		the means to compute the reward value for being at that state.
		Rewards are based on the *_CELL value in the map.
	*/
	public static class Reward_manager implements RewardFunction {
		int[][] map;			// the map with info that we base reward on

		// construction; just stash the map
		public Reward_manager( int[][] map ) {
			this.map = map;
		}

		/*
			Return the reward for being in the given state.
		*/
		@Override
		public double reward( State s, Action a, State sprime ) {
			int x;		// current location
			int y;

			x = (Integer) s.get( VAR_X );		// current position
			y = (Integer) s.get( VAR_Y );

			if( x < 0  || x > map.length ) {
				return -1;						// shouldn't happen, but doesn't hurt to be parinoid
			}
			if( y < 0  || y > map[0].length ) {
				return -1;						// shouldn't happen, but doesn't hurt to be parinoid
			}

			switch( map[x][y] ) {
				case GOAL_CELL:
					return 100.0;

				case NEG_EXTRACT_CELL:			// extraction sell to be avided has high cost
					return -100.0;

				case COSTLY_CELL:				// less desirable to be here
					return -0.75;

				default:						// clear or undefined -- no penality
					return -0.5;
			}

			// unreachable
		}
	}

	// ---------------------------------------------------------------------------
	/*
		Provides the ability to indicate when the agent has reached a termination
		(extraction) point. A copy of the map is provided as there may be more
		than a single termination point which can be reached.
	*/
	public static class Terminator implements TerminalFunction {
		int[][] map;	

		public Terminator( int[][] map ) {
			this.map = map;
		}

		/*
			Return true if the current position of state in the map is on an extraction
			point; false otherwise.
		*/
		@Override
		public boolean isTerminal( State s ) {
			int x;		// current location
			int y;

			x = (Integer) s.get( VAR_X );		// current position
			y = (Integer) s.get( VAR_Y );

			if( x < 0  || x > map.length ) {
				return false;						// shouldn't happen, but doesn't hurt to be parinoid
			}
			if( y < 0  || y > map[0].length ) {
				return false;						// shouldn't happen, but doesn't hurt to be parinoid
			}

			switch( map[x][y] ) {
				case GOAL_CELL:
					return true;


				case NEG_EXTRACT_CELL:			// extraction sell to be avided has high cost
					return true;

				default:						// all other cells are not an extraction point
					return false;
			}

			// unreachable
		}
	}

	// ==============================================================================
	public static void main(String [] args){
		Corn_maze gen;							// the universe as we'll know it
		Corn_maze_runner driver;				// drives analyses
		State initial_state;
		Boolean interactive = false;			// -i turns on
		int argc;
		SADomain domain;
		int[] iter_set = { 100, 100, 100 };		// -i i,p,e can be supplied; this is that set
		boolean show_policy = false;
		boolean run_policy = false;
		boolean run_value = false;
		boolean run_qlearner = false;
		double gamma = 0.99;					// discount (-g)
		double epsilon = 0.1;					// epsilon for qlearning
		int startx = 0;
		int starty = 0;
		String qexp_opt = "egreedy";			// command line exploration string
		int qexp = Corn_maze_runner.EGREEDY;	// exploration type to pass to qlearn
		boolean dump = false;

		argc = 0;
		while( argc < args.length && args[argc].charAt( 0 ) == '-' ) {      // args give some control on command line
			switch( args[argc].charAt( 1 ) ) {
				case '?':
					System.out.printf( "usage: java -cp <classpath> [-d] [-e epsilon] [-E exp-type] [-g gamma] [-i iters] [-P] [-s] [-V] Corn_maze [{easy|city|factory}]\n" );
					System.out.printf( "\t-d  - causes some map info to be dumpped to stderr (useful for plotting)\n" );
					System.out.printf( "\t-E  - qlearn exploration technique; one of: boltz, egreedy, greedyq\n" );
					System.out.printf( "\t-i i0,i1,i2 \n" );
					System.out.printf( "\t-I  - show the universe interactively and do not run expriments\n" );
					System.out.printf( "\t-P  - run the policy iteration experiment\n" );
					System.out.printf( "\t-s  - show the resulting policy and value maps after experiments\n" );
					System.out.printf( "\t-Q  - run the q-learner experiment\n" );
					System.out.printf( "\t-V  - run the value iteration experiment\n" );
					System.exit( 0 );
					break;

				case 'e':	
					epsilon = atof( args[argc+1] );
					if( epsilon < 0.01 ) {
						epsilon = 0.01;
					}
					argc++;
					break;

				case 'E':
					qexp_opt = args[argc+1];
					argc++;
					break;

				case 'd':
					dump = true;
					break;

				case 'g':
					gamma = atof( args[argc+1] );
					if( gamma >= 1 ) {
						gamma = .99;
					} else {
						if( gamma < 0 ) {
							gamma = 0;
						}
					}
					argc++;
					break;
			
				case 'i':
					iter_set = get_vset( args[argc+1], 100 );				
					argc++;
					break;

				case 'I':           // interactive -- show the viewer
					interactive = true;
					break;

				case 'P':
					run_policy = true;
					break;

				case 's':
					show_policy = true;
					break;

				case 'Q':
					run_qlearner = true;
					break;
	
				case 'V':
					run_value = true;
					break;
	
				default:
					System.out.printf( "unrecognised command line option; %s: use -? for usage\n", args[argc]  );
					System.exit( 1 );
					break;
			}

			argc++;
		}

		gen = new Corn_maze();					// create
		gen.select_map( EASY_MAP, !(args.length > argc) );				// default to easy
		if( args.length > argc ) {					// select map based on command line
			if( args[argc].equals( "hard" ) ) {
				gen.select_map( HARD_MAP, dump );
			} else {
				if( args[argc].equals( "factory" ) ) {
					gen.select_map( FACTORY_MAP, dump );
				} else {
					if( args[argc].equals( "city" ) ) {
						gen.select_map( CITY_MAP, dump );
						startx = 23;
						starty = 0;
					}
				} 
			}
		}

		domain = gen.generateDomain();
		initial_state = new Corn_maze_state( startx, starty );		// future take start from command line

		SimulatedEnvironment env = new SimulatedEnvironment( domain, initial_state );

		if( interactive ) {
			VisualExplorer exp;
			Visualizer v;

			v = gen.getVisualizer();
			exp = new VisualExplorer(domain, env, v);

			exp.addKeyAction( "k", ACTION_NORTH, "" );		// um VI commands are better :)
			exp.addKeyAction( "j", ACTION_SOUTH, "" );
			exp.addKeyAction( "l", ACTION_EAST, "" );
			exp.addKeyAction( "h", ACTION_WEST, "" );

			exp.initGUI();		// this doesn't block so ANYTHING below this executes (fail)
		} else {
			double[][] last_v = null;
			double[][] this_v = null;
			double[][] vi_values = null;
			double[][] pi_values = null;
			int i;

			driver = new Corn_maze_runner( gen, domain );

			if( run_value ) {
				System.err.printf( "running value iteration experiments...\n" );
				if( show_policy ) {
					driver.run_va( iter_set, initial_state, gamma, show_policy );
				} else {
					for( i = 1; i < 150; i++ ) {			// run to see where we converge
						iter_set[0] = 100;
						vi_values = driver.run_va( iter_set, initial_state, gamma, false );		// run for compar with pi later

						iter_set[0] = i;
						this_v = driver.run_va( iter_set, initial_state, gamma, false );
						if( i > 1 ) {
							if( Corn_maze_runner.values_match( last_v, this_v ) ) {
								System.err.printf( "converged: %d\n", i );
								break;
							}
						}
						last_v = this_v;
					}
	
					driver.print_values( last_v );
					//vi_values = last_v;
				}

			}

			if( run_policy ) {
				System.err.printf( "running policy iteration experiment\n" );
				if( show_policy ) {
					driver.run_pa( iter_set, initial_state, gamma, show_policy );
				} else {
					iter_set[0] = 100;
					iter_set[1] = 100;
					iter_set[2] = 100;
					pi_values = driver.run_pa( iter_set, initial_state, gamma, show_policy );	// for compare later

					for( i = 1; i < 150; i++ ) {			// run to see where we converge
						iter_set[0] = i;
						iter_set[1] = i;
						iter_set[2] = i;
						this_v = driver.run_pa( iter_set, initial_state, gamma, show_policy );
						if( i > 1 ) {
							if( Corn_maze_runner.values_match( last_v, this_v ) ) {
								System.err.printf( "converged: %d\n", i );
								break;
							}
						}
						last_v = this_v;
					}
	
					driver.print_values( last_v );
					//pi_values = last_v;
				}
			}

			if( run_qlearner ) {
				switch( qexp_opt ) {							// convert command line option to const
					case "bpltzman":
						// fallthrough
					case "boltz":
						qexp = Corn_maze_runner.BOLTZ;
						break;

					case "egreedy":				// defaault
						qexp = Corn_maze_runner.EGREEDY;		// set above, but for completeness
						break;

					case "egmax":
						qexp = Corn_maze_runner.EGMAX;
						break;

					case "greedyd":
						qexp = Corn_maze_runner.GREEDYD;
						break;

					case "greedyq":
						qexp = Corn_maze_runner.GREEDYQ;
						break;

					default:
						qexp = Corn_maze_runner.UNKNOWN;		// will run default and give warning
						break;
				}

				System.err.printf( "running q-learner\n" );
				driver.run_ql( iter_set, initial_state, gamma, show_policy, qexp, epsilon );
			}

			if( pi_values != null && vi_values != null ) {
				System.err.printf( "policy-value difference = %.04f\n",  Corn_maze_runner.values_diff( pi_values, vi_values ) );
				if( Corn_maze_runner.values_match( pi_values, vi_values ) ) {
					System.err.printf( "policy and value iteration methods converged to same value\n" );
				} else {
					System.err.printf( "policy and value iteration methods did NOT converge to same value\n" );
				}
			}
		}
	}
}
				
