
/*
	Mnemonic:	Kcolour.java
	Abstract:	Implements a simplistic K-colouring fitness function.

				There is no error checking on the graph and in good java form a stack
				dump will spew forth if an attempt to reference a node in the graph
				that isn't there.  

				This is original code for the project, based on Abagail optimisation
				examples.

	Author:		Edward Scott Daniels edaniels7@gatech.edu
	Date:		16 February 2019

*/

import java.util.Arrays;

import opt.EvaluationFunction;
import util.linalg.Vector;
import shared.Instance;

public class Kcolour implements EvaluationFunction {
	private class Node {
		int colour;
		Node[] neighbours;
		int nidx;
		int id;

		Node( int id ) {
			neighbours = new Node[100];		// max neighbours
			nidx = 0;
			this.id = id;
		}

		int get_colour( ) {
			return colour;
		}

		void set_colour( int new_colour ) {
			colour = new_colour;
		}

		void add_neighbour( Node neighbour ) {
			neighbours[this.nidx++] = neighbour;		// this will crash if out of bounds and for now that's ok
		}

		/*
			Checks all neighbours. Returns 1 if any neighbour has a matching
			colour as has been assigned to this node. Returns 0 if no 
			neighbour shares the colour.
		*/
		int test_neighbours( ) {
			int i;
	
			for( i = 0; i < nidx; i++ ) {
				if( neighbours[i].get_colour( ) == colour ) {
					return 1;
				}
			}

			return 0;
		}
	}

	private class Graph {
		Node root;
		Node[] nodes;
		int nidx;

		public Graph( int nnodes ) {
			nodes = new Node[nnodes];
			nidx = 0;
		}

		void add_node( int id ) {
			nodes[nidx] = new Node( id );
			nidx++;
		}

		/*
			Connect two nodes, or in other words, make one node the neighbour of 
			another node.
		*/
		void conn_node( int nnum, int neighbour ) {
			nodes[nnum].add_neighbour( this.nodes[neighbour] );
		}

		void colour_node( int nnum, int colour ) {
			nodes[nnum].set_colour( colour );
		}

		int nnodes() {
			return nnodes;
		}

		/* 
			Checks to see if any nodes have the same colour as one of their
			neighbours.  Returns the number of nodes which do NOT have a
			neighbour with a duplicate colour.
		*/
		int test_nodes(  ) {
			int count = 0;
			int i;

			for( i = 0; i < nidx; i++ ) {
				count += nodes[i].test_neighbours( );
			}

			return nidx - count;
		}
	}

	int nnodes = 0;
	Kcolour.Graph g;
	int best = 0;

	/*
		The constructor sets up a basic graph with n nodes. If thought of being 
		positioned in a circle, every node is 'forward connected' to the 
		nodes which are at every other position stopping short of connecting
		back to the first node.  For a graph size 10, this gives two 5 node
		disjoint graphs with the first node of each connected with 5 edges
		meaning that the min-K is 5.  This is done on purpose so that we know
		without a doubt what the min value is so that we can determine which
		randomisers find it.
	*/
    public Kcolour( int n ) {
		int i;
		int j;

		g = new Graph( n );

		if( n < 3 ) {
			nnodes = 3;
		} else {
			nnodes = n;
		}

		for( i = 0; i < nnodes; i++ ) {			// create the network with n nodes
			g.add_node( i );
		}

		for( i = 0; i < nnodes-1; i++ ) {
			for( j = i+2; j < nnodes; j += 2 ) {
				g.conn_node( i, j );			// connect i and j
			}
		}
    }

	/*
		This is the fitness function. Returns the value of the function.
		We take the array given and colour the network with it. Then
		we compute the number of nodes which did NOT have a neighbour
		with the same colour (max).
	*/
    public double value( Instance d ) {
        Vector data;
		int i;
		int right = 0;

		data  = d.getData();
		for( i = 0; i < data.size(); i++ ) {
			g.colour_node( i, (int) data.get( i ) );		// colour each node
		}

/*
		right = g.test_nodes();
		if( right > best ) { 
			best = right;
			System.out.printf( "best = %d\n", right );
		}
*/
		
			
/*
		if( g.test_nodes() == 8 ) {
			for( i = 0; i < data.size(); i++ ) {
				System.out.printf( "%d ", (int) data.get( i ) );
			}
				System.out.printf( "\n" );
		}
*/

		return g.test_nodes();		// return num 'good' nodes
    }
}
