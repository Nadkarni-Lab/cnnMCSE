"""Utility functions for calculate convergence sample estimation from loss curves. 
"""
import numpy as np

from sklearn.datasets import make_classification
from scipy.interpolate import UnivariateSpline

def get_derivative(A3_loss_raw:list, sample_sizes:list):
    """Get the derivative of the loss function. 

    Args:
        A3_loss_raw (list): _description_
        sample_sizes (list): _description_

    Returns:
        spline, spline, spline: spline, 1st and second derivative. 
    """
    log_sample_sizes = np.log(sample_sizes)
    A3_loss_mean = [np.mean(a3_loss) for a3_loss in A3_loss_raw]
    y_spl = UnivariateSpline(log_sample_sizes, A3_loss_mean,s=0,k=2)
    y_spl_1d = y_spl.derivative(n=1)
    y_spl_2d = y_spl.derivative(n=2)
    
    return y_spl, y_spl_1d, y_spl_2d

def get_inflection_point(spl, spl_2D, sample_sizes):
    """Get the inflection point and the errors. 

    Args:
        spl (spline): Spline. 
        spl_2D (spline): Second derivative
        sample_sizes (list): List of sample sizes. 

    Returns:
        peak_A3, error_down, error_up: peak and the error up and error down. 
    """
    log_sample_sizes = np.log(sample_sizes)
    sample_size_peak = np.argmax(spl_2D(log_sample_sizes))
    peak_A3 = spl(log_sample_sizes[sample_size_peak])
    error_down = spl(log_sample_sizes[sample_size_peak+1])
    error_up   = spl(log_sample_sizes[sample_size_peak-1])
    return peak_A3, error_down, error_up



def generate_sample_sizes(max_sample_size : int = 5000, log_scale: int = 2, min_sample_size: int = 64, absolute_scale = False):
    sample_size_list = list()

    if(absolute_scale == False):
        sample_size = int(max_sample_size)
        while sample_size > min_sample_size:
            sample_size_list.append(sample_size)
            sample_size = int(sample_size / log_scale)
        sample_size_list.append(min_sample_size)
    
    else:
        for sample_size in range(min_sample_size, max_sample_size, absolute_scale):
            sample_size_list.append(sample_size)
        sample_size_list.append(max_sample_size)
    sample_size_list.sort()
    print(sample_size_list)
    return sample_size_list
