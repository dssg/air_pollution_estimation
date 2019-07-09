import numpy as np 


def time_series_smoother(array:np.ndarray, method:str, 
					     window_size = 10, poly_degree = 10):    
	"""Array is in shape (vehicle, ious over time)
   		Smoothing function for time series data 
	"""  
	assert len(array.shape) == 2

	#smoothing options
	if method == "kalman_filter": 
	    return 0

	# fit a polynomial 
	elif method == "polynomial": 
		poly_fit = np.zeros(array.shape)
		x_inds = np.arange(array.shape[1]) # num frames 
		num_series = array.shape[0]
		for i in range(num_series): #iterate over axis 1 of array
			poly_coeffs = np.polyfit(x = x_inds, 
									y = array[i,:], 
									deg = poly_degree)
			poly_fit[:,i] = poly_coeffs @ x_inds
	    return poly_fit

	elif method == "moving_avg":
	    #return cumulative sum 
	    mv_avg = np.cumsum(array, dtype=float, axis = 1)
	    mv_avg[:,window_size:] = mv_avg[:,window_size:] - mv_avg[:,:-window_size]
	    return mv_avg[:,window_size - 1:] / window_size
