from logistic import logistic_map, iterate_f
import numpy as np
import pytest

SEED = 42

@pytest.mark.parametrize("r, x, expected",[(2.2, 0.1, 0.198),(3.4,0.2, 0.544),(1.7,0.75,0.31875)])
def test_logistic_map(r,x, expected):
	output = logistic_map(r,x)
	assert  np.isclose( output,  expected )
	
@pytest.mark.parametrize( "r, x, it, expected", [(2.2, 0.1, 1.0, [0.198]), (3.4, 0.2, 4.0, [0.544, 0.843418, 0.449019, 0.841163] ), (1.7, 0.75, 2, [0.31875, 0.369152])] )
def test_iterate_f( r, x, it, expected ):
	output = iterate_f( r, x, int(it) )
	assert np.allclose( output, expected )
	
	
def test_fuzzing():
	r=1.5
	it = 100
	rand_state = np.random.RandomState(SEED)

	x_vals = rand_state.random(15)
	expected = 1/3

	for item in x_vals:	
		output = iterate_f(r, item, int(it))
		assert np.isclose( output[-1], expected, rtol=1e-05 )
		
	print("random seed =", SEED)
