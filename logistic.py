import numpy as np


def logistic_map(r,x):
	func = r*x*(1-x)
	return func

#def iterate_f( r, x, iterations ):
#	if iterations > 0:
#		f = iterate_f( x, iterations )
#		iterations = iterations - 1
#	else:
#		return logistic_map( r, x )		
		
def iterate_f( r, x, iterations ):

	outs = []

	for i in range( int(iterations) ):
		x = logistic_map( r, x )
		outs.append( x )
	outs = np.array( outs )
	return outs

#r = 2.1
#x = 0.1
#it = 5	
	
#plot_trajectory(it, r, x, "figure")
