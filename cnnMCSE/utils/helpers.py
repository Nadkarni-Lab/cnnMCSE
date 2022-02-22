"""Utility functions for calculate convergence sample estimation from loss curves. 
"""
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

def get_derivative(loss_list:list, sample_sizes:list):
    """Get the derivative of the loss function. 

    Args:
        loss_list (list): _description_
        sample_sizes (list): _description_

    Returns:
        spline, spline, spline: spline, 1st and second derivative. 
    """
    log_sample_sizes = np.log(sample_sizes)
    loss_mean = [np.mean(loss) for loss in loss_list]
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


def estimate_mcse(df:pd.DataFrame, out_metadata_path:str):
    """Method to estimate minimum convergence sample from pandas dataframe. 

    Args:
        df (pd.DataFrame): Pandas dataframe with sample_sizes, estimators, and estimands. 
        out_metadata_path (str): Outfile to store. 
    """

    # generate list of sample sizes. 
    sample_sizes = list(df['sample_size'].unique())

    # generate list of losses (estimator)
    loss_list = list()
    for sample_size in sample_sizes:
        sample_size_df = df[df['sample_size'] == sample_size]
        sample_size_estimators = sample_size_df['estimators'].tolist()
        loss_list.append(loss_list)
    
    # estimate the spline. 
    estimator_spl, estimator_spl_1d, estimator_spl_2d = get_derivative(loss_list=loss_list, sample_sizes=sample_sizes)

    # estimate the peaks and error bars. 
    peak_estimator, error_down, error_up = get_inflection_point(spl=estimator_spl, spl_2D=estimator_spl_2d)

    # generate metadata output file. 
    out_metadata_dict = {}
    out_metadata_dict['mcse'] = peak_estimator
    out_metadata_dict['mcse_error_up'] = error_up
    out_metadata_dict['mcse_error_down'] = error_down
    out_metadata_df = pd.DataFrame(out_metadata_dict)
    out_metadata_df.to_csv(out_metadata_path, sep="\t")




def generate_sample_sizes(max_sample_size : int = 5000, log_scale: int = 2, min_sample_size: int = 64, absolute_scale = False):
    sample_size_list = list()


    if(absolute_scale == False):
        current_sample_size = min_sample_size
        while current_sample_size < max_sample_size:
            sample_size_list.append(current_sample_size)
            current_sample_size = current_sample_size * log_scale
    # if(absolute_scale == False):
    #     sample_size = int(max_sample_size)
    #     while sample_size > min_sample_size:
    #         sample_size_list.append(sample_size)
    #         sample_size = int(sample_size / log_scale)
    #     sample_size_list.append(min_sample_size)
    
    else:
        for sample_size in range(min_sample_size, max_sample_size, absolute_scale):
            sample_size_list.append(sample_size)
        sample_size_list.append(max_sample_size)
    sample_size_list.sort()
    print(sample_size_list)
    return sample_size_list
