// vi: ts=4 sw=4 noet:
/*
	Mnemonic:	Corn_maze_state.java
	Abstract:	This provides state information for any state in
				the grid, and provides operators (e.g. copy) on
				those states. This code taken from the Burlap 
				examples almost as is (with formatting cleanup).
				The one change was to deal with dodgy key values
				passed into the get function.
	Author:		Edward Scott Daniels
	Date:		30 March 2019
*/


import burlap.mdp.core.state.MutableState;
import burlap.mdp.core.state.StateUtilities;
import burlap.mdp.core.state.UnknownKeyException;
import burlap.mdp.core.state.annotations.DeepCopyState;

import java.util.Arrays;
import java.util.List;


@DeepCopyState
public class Corn_maze_state implements MutableState {
	public static final String VAR_X = "x";
	public static final String VAR_Y = "y";

	public int x;
	public int y;

	private final static List<Object> keys = Arrays.<Object>asList(VAR_X, VAR_Y);

	public Corn_maze_state() {
		x = y = 0;
	}

	public Corn_maze_state(int x, int y) {
		this.x = x;
		this.y = y;
	}

	@Override
	public MutableState set( Object variableKey, Object value ) {
		if(variableKey.equals(VAR_X)){
			this.x = StateUtilities.stringOrNumber(value).intValue();
		} else {
			if(variableKey.equals(VAR_Y)) {
				this.y = StateUtilities.stringOrNumber(value).intValue();
			} else {
				throw new UnknownKeyException( variableKey );
			}
		}

		return this;
	}

	public List<Object> variableKeys() {
		return keys;
	}

	/*
		For some odd reason the policy gui painter wants to pass agent:{x|y} instead of
		just x or y.  We accept agent:* as x or y based on the last character.
	*/
	@Override
	public Object get( Object variableKey ) {

		if( variableKey.equals(VAR_X) || variableKey.toString().equals( "agent:x" ) ) {
			return x;
		}

		if( variableKey.equals(VAR_Y) || variableKey.toString().equals( "agent:y" ) ) {
			return y;
		}

		System.err.printf( ">>>> could not handle this key: (%s)\n", variableKey.toString() );
		System.exit( 1 );	// stack dump is useless here, just barf and quit.
		//throw new UnknownKeyException( variableKey );		// this is crappy.

		return -1;		// unreachable with exit above, but java too dumb to see that.
	}

	/*
		For unknown reasons the get() function in State returns Obj and not int.  I can't
		figure out how to convert that into a _real_ value, so here is this. 
	*/
	public int get_int( Object variableKey ) {

		if( variableKey.equals(VAR_X) || variableKey.toString().equals( "agent:x" ) ) {
			return x;
		}

		if( variableKey.equals(VAR_Y) || variableKey.toString().equals( "agent:y" ) ) {
			return y;
		}

		System.err.printf( ">>>> could not handle this key: (%s)\n", variableKey.toString() );
		System.exit( 1 );	// stack dump is useless here, just barf and quit.
		//throw new UnknownKeyException( variableKey );		// this is crappy.

		return -1;		// unreachable with exit above, but java too dumb to see that.
	}

	@Override
	public Corn_maze_state copy() {
		return new Corn_maze_state( x, y );
	}

	@Override
	public String toString() {
		return StateUtilities.stateToString( this );
	}
}
