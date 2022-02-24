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
    y_spl = UnivariateSpline(log_sample_sizes, loss_mean,s=0,k=2)
    y_spl_1d = y_spl.derivative(n=1)
    y_spl_2d = y_spl.derivative(n=2)
    
    return y_spl, y_spl_1d, y_spl_2d

def get_inflection_point(spl, spl_2D, sample_sizes, estimator:bool=True):
    """Get the inflection point and the errors. 

    Args:
        spl (spline): Spline. 
        spl_2D (spline): Second derivative
        sample_sizes (list): List of sample sizes. 
        estimator (bool): Estimator of all 

    Returns:
        peak_A3, error_down, error_up: peak and the error up and error down. 
    """
    log_sample_sizes = np.log(sample_sizes)
    if(estimator == True):
        sample_size_peak = np.argmin((spl_2D(log_sample_sizes)))
    else:
        sample_size_peak = np.argmax((spl_2D(log_sample_sizes)))
    peak = (log_sample_sizes[sample_size_peak])
    error_down = (log_sample_sizes[sample_size_peak+1])
    error_up   = (log_sample_sizes[sample_size_peak-1])
    return peak, error_down, error_up

def get_power(spl, value:float):
    """Get the power at a given value. 

    Args:
        spl (spl): Power curve. 
        value (float): Value to estimate power at

    Returns:
        float: Power estimate. 
    """

    return spl(value)
def estimate_mcse(df:pd.DataFrame, out_metadata_path:str):
    """Method to estimate minimum convergence sample from pandas dataframe. 

    Args:
        df (pd.DataFrame): Pandas dataframe with sample_sizes, estimators, and estimands. 
        out_metadata_path (str): Outfile to store. 
    """

    # generate list of sample sizes. 
    sample_sizes = list(df['sample_size'].unique())
    print(f"Generating sample sizes, {sample_sizes}")

    # generate list of losses (estimator)
    loss_list = list()
    auc_list = list()
    for sample_size in sample_sizes:
        sample_size_df = df[df['sample_size'] == sample_size]
        sample_size_estimators = sample_size_df['estimators'].tolist()
        sample_size_estimands = sample_size_df['estimands'].tolist()
        loss_list.append(sample_size_estimators)
        auc_list.append(sample_size_estimands)
        print(f"Adding losses .... {loss_list}")
        print(f"Adding AUCs .... {auc_list}")
    
    # estimate the spline. 
    print("Estimating the spline.")
    estimator_spl, _, estimator_spl_2d = get_derivative(loss_list=loss_list, sample_sizes=sample_sizes)
    estimand_spl,  _, estimand_spl_2d  = get_derivative(loss_list=auc_list, sample_sizes=sample_sizes)


    # estimate the peaks and error bars. 
    print("Estimating peaks and errors.")
    peak_estimator, error_down, error_up  = get_inflection_point(spl=estimator_spl, spl_2D=estimator_spl_2d, sample_sizes=sample_sizes, estimator = True)
    peak_estimand, estimand_down, estimand_up = get_inflection_point(spl=estimand_spl, spl_2D=estimand_spl_2d, sample_sizes=sample_sizes, estimator = False)

    power_estimator = get_power(spl=estimand_spl, value=peak_estimator)
    power_down  = get_power(spl=estimand_spl, value=error_down), 
    power_up = get_power(spl=estimand_spl, value=error_up)

    # generate metadata output file. 
    out_metadata_dict = {}
    out_metadata_dict['mcse'] = [peak_estimator]
    out_metadata_dict['mcse_error_up'] = [error_up]
    out_metadata_dict['mcse_error_down'] = [error_down]
    out_metadata_dict['mcse_power'] = [power_estimator]
    out_metadata_dict['mcse_power_up'] = [power_up]
    out_metadata_dict['mcse_power_down'] = [power_down]

    out_metadata_dict['mcs'] = [peak_estimand]
    out_metadata_dict['mcs_error_up'] = [estimand_up]
    out_metadata_dict['mcs_error_down'] = [estimand_down]

    out_metadata_df = pd.DataFrame(out_metadata_dict)
    print(out_metadata_df)
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
