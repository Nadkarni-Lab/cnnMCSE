"""Utility functions for calculate convergence sample estimation from loss curves. 
"""
from logging import root
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from cnnMCSE.experiments.sampling import sampling_helper
from cnnMCSE.experiments.complexity import complexity_helper
from cnnMCSE.experiments.labels import labels_helper
from cnnMCSE.mimic.cxr import mimic_helper
from cnnMCSE.mimic.nih import nih_helper
from cnnMCSE.mimic.chexpert import chexpert_helper
from cnnMCSE.experiments.synthetic import synthetic_helper
from cnnMCSE.dissecting.cost import dissecting_helper
import seaborn as sns
import matplotlib.pyplot as plt 

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
    print("Log sample sizes", log_sample_sizes)
    print("Loss mean", loss_mean)
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

def estimate_smcse(df:pd.DataFrame):
    """Method to estimate stratified minimum convergence sample estimations. 

    Args:
        df (pd.DataFrame): _description_
    """

    # get column names
    col_names = list(df.columns)

    out_estimand = estimate_mcse(df)
    out_estimand['label'] = 'all'
    metadata_dfs = [out_estimand]
    # filter to estimands

    estimands_labels = list(df['label'].unique())
    #estimand_cols = [col_name for col_name in col_names if ('estimands__' in col_name)]
    #other_cols = [col_name for col_name in col_names if ('estimands__' not in col_name)]
    for estimand_col in estimands_labels:
        print("Running estimand ...", estimand_col)
        #class_name = estimand_col.split("__")[-1]
        #df_estimand = df[other_cols + [estimand_col]]
        #df_estimand['estimands'] = df[estimand_col]

        df_estimand = df[
            df['label'] == estimand_col
        ]
        df_estimand['estimators'] = df_estimand['s_estimators']
        df_estimand['estimands'] = df_estimand['s_estimands']

        out_estimand = estimate_mcse(df_estimand)
        out_estimand['label'] = estimand_col
        #out_estimand.columns = [col_df + '__' + str(estimand_col) for col_df in out_estimand.columns]
        print(out_estimand)
        metadata_dfs.append(out_estimand)
    

    print(len(metadata_dfs))
    metadata_df = pd.concat(metadata_dfs, axis=0)
    return metadata_df




def estimate_mcse(df:pd.DataFrame):
    """Method to estimate minimum convergence sample from pandas dataframe. 

    Args:
        df (pd.DataFrame): Pandas dataframe with sample_sizes, estimators, and estimands. 
        out_metadata_path (str): Outfile to store. 
    """

    # generate list of sample sizes. 
    print("Importing dataframe.")


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
    out_metadata_dict['mcse_power_down'] = [power_down[0]]

    out_metadata_dict['mcs'] = [peak_estimand]
    out_metadata_dict['mcs_error_up'] = [estimand_up]
    out_metadata_dict['mcs_error_down'] = [estimand_down]

    out_metadata_df = pd.DataFrame(out_metadata_dict)
    return out_metadata_df
    #print(out_metadata_df)
    #out_metadata_df.to_csv(out_metadata_path, sep="\t")

def get_mcse_discrete(fcn_data, current_label, current_dataset, current_bootstrap):
    #print(fcn_data.dtypes)
    #print('Current_label', type(current_label))
    #print('dataset', type(current_dataset))
    #print('bootstrap', type(current_bootstrap))


    test = fcn_data[(fcn_data["label"] == current_label) & 
                    (fcn_data["dataset"] == current_dataset) &
                    (fcn_data["bootstrap"] == current_bootstrap)]
    test["log_sample_size"] = np.log2(test["sample_size"])
    test = test.groupby("log_sample_size").agg({'s_estimators': ['mean']})
    test = test.reset_index()


    mcse_index = np.argmax(np.diff(test["s_estimators"]["mean"], n=2)) + 2
    mcse_log_sample_size = test["log_sample_size"][mcse_index]
    return mcse_log_sample_size


def generate_sample_sizes(max_sample_size : int = 5000, log_scale: int = 2, min_sample_size: int = 64, absolute_scale = None):
    sample_size_list = list()

    # print('Absolute scale', absolute_scale)
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
    # print(sample_size_list)
    return sample_size_list

def post_process_estimands(df:pd.DataFrame, 
    demographic:str, 
    n_bootstraps:int, 
    outcome_pre:str, 
    outcome_post:str, 
    demographic_pre:str, 
    demographic_post:str):
    """Method to post-process the estimands to measure bias. 

    Args:
        df (pd.DataFrame): The dataframe output by the mitigate_bias_customs function. 
        demographic (str): Demographic to evaluate on.
        n_bootstraps (int): Number of bootstraps. 
        outcome_pre (str): Outcome selected for pre-mitigation. 
        outcome_post (str): Outcome selected for post-mitigation. 
        demographic_pre (str): Demographic trained on pre-mitigation. 
        demographic_post (str): Demographic trained on post mitigation. 

    Returns:
        pd.DataFrame: Effect of intervention on metric. 
    """
    df2 = df[
        (df['demographics_test'] == demographic) &
        (df['sample_size'] == df['sample_size'].max())
    ]


    df2 = df2.groupby(["sample_size", "demographics_train", "outcome_train"]).agg({
            "estimands": ["mean", "std"]
    })
    df2["auc_mean"] = df2["estimands"]["mean"]
    df2["auc_se"]   = df2["estimands"]["std"] / n_bootstraps

    df2.reset_index(inplace=True)
    df2.drop(["estimands", "sample_size"], axis=1, inplace=True)

    pre =  df2[
        ((df2['demographics_train'] == demographic_pre)  & (df2['outcome_train'] == outcome_pre))]
    pre['intervention'] = 'Pre'
    post =  df2[
        ((df2['demographics_train'] == demographic_post)  & (df2['outcome_train'] == outcome_post))]
    post['intervention'] = 'Post'

    df3 = pd.concat([pre, post], axis=0)
    df3 = df3[['auc_mean', 'auc_se', 'intervention']]
    return df3


def visualize_changes(data):
    ax = sns.pointplot('demographics', 'mean_mcse', hue='label',
    data=data, dodge=True, join=False, ci=None, scale=0.75)

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)
    errors = data['sd_mcse']
    colors = ['steelblue']*3 + ['coral']*3 
    ax.errorbar(x_coords, y_coords, yerr=errors,
        ecolor=colors, fmt=' ', zorder=-1)
    ax.set_ylabel("AEq")
    plt.show()

def experiment_helper(experiment:str, dataset:str, root_dir:str, start_seed:int, tl_transforms:bool=False, input_dim:int=None):
    print("Experiment", experiment)
    print("Dataset", dataset)
    print("Root-dir", root_dir)
    if(experiment == "sampling"):
        print("Getting sampling experiment.")
        return sampling_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    
    elif(experiment == "complexity"):
        print("Getting complexity experiment.")
        return complexity_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms)

    elif(experiment == "label"):
        print("Getting label bias experiment.")
        return labels_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms, start_seed=start_seed)
    
    elif(experiment == "MIMIC"):
        return mimic_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms)
    
    elif(experiment == "NIH"):
        return nih_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms)
    
    elif(experiment == "CHEXPERT"):
        return chexpert_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms)
    
    elif(experiment == "synthetic"):
        return synthetic_helper(dataset=dataset, root_dir=root_dir, tl_transforms=tl_transforms, input_dim=input_dim)

    elif(experiment == "dissecting"):
        return dissecting_helper(dataset=dataset, root_dir=root_dir, start_seed=start_seed)
    
    else:
        return None
    



    pass