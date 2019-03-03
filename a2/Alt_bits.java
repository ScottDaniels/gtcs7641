
/*
	Mnemonic:	Alt_bits.java
	Abstract:	Implements a simplistic alternating bits fitness function.
				There are two optimal patterns:
					101010...
					010101...

				The fitness function returns the number of bits in the correct
				position based on the first.

				The range pass to the various optimisation algorithms should be
				an n byte array with all elements set to 2.

				This is original code for the project, based on the examples
				in the Abagail optimisation.

	Author:		Edward Scott Daniels  edaniels7@gatech.edu
	Date:		25 February 2019
*/

import java.util.Arrays;

import opt.EvaluationFunction;
import util.linalg.Vector;
import shared.Instance;


public class Alt_bits implements EvaluationFunction {
	/*
		This is the fitness function. Returns the value of the data
		passed in which is the number of bits in the correct position. 
		The maximum value is the size of the array. There are two possible
		bit strings that give the maximum value.
	*/
    public double value( Instance d ) {
        Vector data;
		int i;
		int[] expect;		// value expected at even/odd positions in the data
		int value = 0;		// the value of the data passed in

		expect = new int[2];
		data  = d.getData();
		if( data.get( 0 ) == 0 ) {
			expect[0] = 0;
			expect[1] = 1;
		} else {
			expect[0] = 1;
			expect[1] = 0;
		}
		for( i = 0; i < data.size(); i++ ) {
			if( (int) data.get( i ) == expect[i%2] ) {
				value++;
			}
		}

		return value;
    }
}
