import numpy as np


def time_series_smoother(array: np.ndarray, method: str,
                         window_size=25, poly_degree=10):
    """Smoothing function for time series data 

    Keyword arguments

    array -- in shape (vehicle, ious over time)
    method -- see code to see which smoothing methods are currently supported 
    window_size -- parameter for the moving average method
    poly_degree -- parameter fo the polynomial method 
    """
    assert len(array.shape) == 2

    # smoothing options
    if method == "kalman_filter":
        # TODO: explore this smoothing option in the future
        pass

    # fit a polynomial
    elif method == "polynomial":
        smoothed_array = np.zeros(array.shape)
        x_inds = np.arange(array.shape[1])  # num frames
        powers = np.arange(poly_degree + 1)  # power array

        num_series = array.shape[0]
        for i in range(num_series):  # iterate over individual time series in array
            # get non-na and non-inf values from the time series and corresp indices
            time_series = array[i, :]
            na_mask, inf_mask = np.isnan(time_series), np.isfinite(time_series)
            x_inds_filt = x_inds[~na_mask & inf_mask]
            time_series_filt = time_series[~na_mask & inf_mask]

            # only fit poly to non-na and non-inf values
            poly_coeffs = np.flip(np.polyfit(x=x_inds_filt,
                                             y=time_series_filt,
                                             deg=poly_degree))
            # copy each entry in x_inds_filt poly_degree number of times, so that we can predict on it
            # using the newly fit polynomial
            x_inds_filt_rpted = np.tile(x_inds_filt, 
                                       (poly_degree+1, 1)).transpose()
            pow_x = np.power(x_inds_filt_rpted, powers).transpose()

            preds = poly_coeffs @ pow_x
            # fill na locations with zeros
            preds_with_zeros = np.zeros(array.shape[1])
            np.put(preds_with_zeros, x_inds_filt, preds)
            # insert 0 where not x inds are
            smoothed_array[i, :] = preds_with_zeros

    elif method == "moving_avg":
        mv_avg = np.cumsum(array, dtype=float, axis=1)
        # return cumulative sum
        mv_avg[:, window_size:] = mv_avg[:, window_size:] - \
            mv_avg[:, :-window_size]
        smoothed_array = mv_avg[:, window_size - 1:] / window_size

    return smoothed_array
