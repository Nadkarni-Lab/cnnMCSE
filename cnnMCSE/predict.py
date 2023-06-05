"""Generate estimators and estimands for sample size estimation with convolutional neural networks. 
"""
import gc
import os
import torch
from pyexpat import model
import pandas as pd
import itertools

from cnnMCSE.dataloader import dataloader_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.helpers import experiment_helper, generate_sample_sizes, get_derivative, get_inflection_point, get_mcse_discrete
from cnnMCSE.mcse import get_estimators, get_estimands
from cnnMCSE.utils.helpers import estimate_mcse, estimate_smcse, experiment_helper
from cnnMCSE.experiments.sampling import sampling_helper
from cnnMCSE.experiments.custom import custom_helper

def predict_loop(
    datasets:str,
    models:str,
    root_dir:str,
    batch_size:int,
    n_workers:int,
    max_sample_size:int,
    log_scale:int,
    min_sample_size:int,
    absolute_scale:bool,
    n_bootstraps:int,
    initial_weights_dir:str, 
    out_data_path:str,
    n_epochs:int=1,
    state_dict_dir:str=None,
    out_metadata_path:str=None, 
    start_seed:int = 42,
    shuffle:bool=False,
    num_workers:int=4,
    zoo_models:str=None,
    metric_type:str="AUC",
    frequency:bool=False,
    stratified:bool=False
    ):

    # initialize datasets
    dataset_list = datasets.split(",")
    print(f"Testing datasets... {dataset_list}")

    # initialize models
    models = models.split(",") 
    print(f"Testing models... {models}")
    estimator, estimand = models
    estimator, initial_estimator_weights_path = model_helper(model=estimator, initial_weights_dir=initial_weights_dir)
    estimand, initial_estimand_weights_path = model_helper(model=estimand, initial_weights_dir=initial_weights_dir)

    # initialize transfer learning zoo. 
    using_pretrained = (zoo_models != None)
    if(using_pretrained): 
        zoo_model_list = zoo_models.split(",")
        print(f"Using transfer learning base... {zoo_model_list}")
        torch.hub.set_dir(initial_weights_dir)
    else: 
        print("No transfer learning base.")
        zoo_model_list = [None]


    dfs = list()
    for current_dataset in dataset_list:
        trainset, testset = dataloader_helper(current_dataset, root_dir, tl_transforms=using_pretrained)
        sample_sizes = generate_sample_sizes(
            max_sample_size=max_sample_size, 
            log_scale=log_scale, 
            min_sample_size=min_sample_size, 
            absolute_scale=absolute_scale
        )
        print(f"Running sample sizes... {sample_sizes}")
        for zoo_model in zoo_model_list:
            print(f"Using transfer learning model... {zoo_model}")
            for sample_size in sample_sizes:

                #---- estimation block
                estimators = get_estimators(
                    model = estimator,
                    training_data = trainset,
                    sample_size = sample_size,
                    batch_size = batch_size,
                    bootstraps = n_bootstraps,
                    start_seed = start_seed,
                    shuffle = shuffle,
                    initial_weights=initial_estimator_weights_path,
                    num_workers=num_workers, 
                    zoo_model=zoo_model,
                    frequency=frequency,
                    stratified=stratified,
                    n_epochs=n_epochs
                )
                

                estimands = get_estimands(
                    model = estimand,
                    training_data = trainset,
                    validation_data = testset,
                    sample_size = sample_size,
                    batch_size=batch_size,
                    bootstraps=n_bootstraps,
                    start_seed=start_seed,
                    shuffle=shuffle,
                    metric_type=metric_type,
                    initial_weights=initial_estimand_weights_path,
                    num_workers=num_workers,
                    zoo_model=zoo_model,
                    n_epochs=n_epochs
                )

                #--- Logging block. 
                print("Logging results... ")
                df_dict = {}
                if(frequency == False):
                    df_dict['estimators']   = estimators

                if(metric_type == "AUC"):
                    df_dict['estimands']    = estimands
                df_dict['bootstrap']    = [i for i in range(n_bootstraps)]
                df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
                df_dict['backend']      = [f'{str(zoo_model)}' for i in range(n_bootstraps)]
                df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
                df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
                df_dict['dataset']      = [f'{current_dataset}' for i in range(n_bootstraps)]
                df = pd.DataFrame(df_dict)
                print(df)
                print(estimands)
                print(estimators)
                if(metric_type == "sAUC"):
                    if(frequency):
                        outputs = estimators.merge(estimands, on=['label', 'bootstrap'])
                        print(outputs)
                        # outputs = pd.concat([estimands, estimators], axis=1)
                        df_merged = outputs.merge(df, on=['bootstrap', 'sample_size'])
                        # df_merged = pd.concat([df, outputs], axis=1)
                    else:
                        df_merged = pd.concat([df, estimators], axis=1)

                dfs.append(df_merged)

                gc.collect()
    df = pd.concat(dfs)
    df.to_csv(out_data_path, sep="\t", index=False)
    

    ## Evaluating Models.
    meta_dfs = list() 
    for current_dataset in dataset_list:
        for zoo_model in zoo_model_list:
            current_df = df[
                (df['backend'] == str(zoo_model)) &
                (df['dataset'] == str(current_dataset))
            ]
            if(metric_type == "AUC"):
                meta_df              = estimate_mcse(current_df)
            else:
                meta_df              = estimate_smcse(current_df)
            meta_df['bootstrap'] = n_bootstraps
            meta_df['backend']   = zoo_model
            meta_df['dataset']   = current_dataset
            meta_df['estimator'] = estimator
            meta_df['estimand']  = estimand
            meta_dfs.append(meta_df)
    
    meta_df = pd.concat(meta_dfs)
    meta_df.to_csv(out_metadata_path, sep='\t', index=False)

    print("Complete")

def experiment_loop(
    datasets:str,
    models:str,
    root_dir:str,
    batch_size:int,
    n_epochs:int,
    n_workers:int,
    max_sample_size:int,
    log_scale:int,
    min_sample_size:int,
    absolute_scale:bool,
    n_bootstraps:int,
    initial_weights_dir:str, 
    out_data_path:str,
    state_dict_dir:str=None,
    out_metadata_path:str=None, 
    start_seed:int = 42,
    shuffle:bool=False,
    num_workers:int=4,
    zoo_models:str=None,
    metric_type:str="AUC",
    frequency:bool=False,
    stratified:bool=False,
    experiment:str=None,
    input_dim:int=None
    ):

    # initialize datasets
    dataset_list = datasets.split(",")
    print(f"Testing datasets... {dataset_list}")

    # initialize models
    models = models.split(",") 
    print(f"Testing models... {models}")
    estimator, estimand = models
    estimator, initial_estimator_weights_path = model_helper(model=estimator, initial_weights_dir=initial_weights_dir)
    estimand, initial_estimand_weights_path = model_helper(model=estimand, initial_weights_dir=initial_weights_dir)

    # initialize transfer learning zoo. 
    using_pretrained = (zoo_models != None)
    if(using_pretrained): 
        zoo_model_list = zoo_models.split(",")
        print(f"Using transfer learning base... {zoo_model_list}")
        torch.hub.set_dir(initial_weights_dir)
    else: 
        print("No transfer learning base.")
        zoo_model_list = [None]


    dfs = list()

    # initialize dataset dictionary
    for current_dataset in dataset_list:
        dataset_dict = experiment_helper(experiment=experiment, dataset=current_dataset, root_dir=root_dir, tl_transforms=using_pretrained, start_seed=start_seed)
        print(dataset_dict)
        #dataset_dict = sampling_helper(dataset=current_dataset, root_dir=root_dir)

        # match training and testing
        train_test_match = dataset_dict['train_test_match']

        for trainset, testset in train_test_match:
            print(f"Running trainset - {trainset}, testset - {testset}")
            sample_sizes = generate_sample_sizes(
                max_sample_size=max_sample_size, 
                log_scale=log_scale, 
                min_sample_size=min_sample_size, 
                absolute_scale=absolute_scale
            )
            print(f"Running sample sizes... {sample_sizes}")
            for zoo_model in zoo_model_list:
                print(f"Using transfer learning model... {zoo_model}")
                for sample_size in sample_sizes:

                    #---- estimation block
                    estimators = get_estimators(
                        model = estimator,
                        training_data = dataset_dict[trainset],
                        sample_size = sample_size,
                        batch_size = batch_size,
                        bootstraps = n_bootstraps,
                        start_seed = start_seed,
                        shuffle = shuffle,
                        initial_weights=initial_estimator_weights_path,
                        num_workers=n_workers, 
                        zoo_model=zoo_model,
                        frequency=frequency,
                        stratified=stratified
                    )
                    

                    estimands = get_estimands(
                        model = estimand,
                        training_data = dataset_dict[trainset],
                        validation_data = dataset_dict[testset],
                        sample_size = sample_size,
                        batch_size=batch_size,
                        bootstraps=n_bootstraps,
                        start_seed=start_seed,
                        shuffle=shuffle,
                        metric_type=metric_type,
                        initial_weights=initial_estimand_weights_path,
                        num_workers=n_workers,
                        zoo_model=zoo_model
                    )

                    #--- Logging block. 
                    print("Logging results... ")
                    df_dict = {}
                    if(frequency == False):
                        df_dict['estimators']   = estimators

                    if(metric_type == "AUC"):
                        df_dict['estimands']    = estimands
                    df_dict['bootstrap']    = [i for i in range(n_bootstraps)]
                    df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
                    df_dict['backend']      = [f'{str(zoo_model)}' for i in range(n_bootstraps)]
                    df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
                    df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
                    df_dict['trainset']     = [f'{trainset}' for i in range(n_bootstraps)]
                    df_dict['testset']      = [f'{testset}' for i in range(n_bootstraps)]
                    df_dict['dataset']      = [f'{trainset}_{testset}_{current_dataset}' for i in range(n_bootstraps)]
                    df = pd.DataFrame(df_dict)
                    print(df)
                    print(estimands)
                    print(estimators)
                    if(metric_type == "sAUC"):
                        if(frequency):
                            outputs = estimators.merge(estimands, on=['label', 'bootstrap'])
                            print(outputs)
                            # outputs = pd.concat([estimands, estimators], axis=1)
                            df_merged = outputs.merge(df, on=['bootstrap', 'sample_size'])
                            # df_merged = pd.concat([df, outputs], axis=1)
                        else:
                            df_merged = pd.concat([df, estimators], axis=1)

                    dfs.append(df_merged)

                    gc.collect()
    df = pd.concat(dfs)
    df.to_csv(out_data_path, sep="\t", index=False)
    

    ## Evaluating Models.
    train_test_data_list = list(df['dataset'].unique())
    meta_dfs = list() 
    for current_dataset in train_test_data_list:
        for zoo_model in zoo_model_list:
            current_df = df[
                (df['backend'] == str(zoo_model)) &
                (df['dataset'] == str(current_dataset))
            ]
            if(metric_type == "AUC"):
                meta_df              = estimate_mcse(current_df)
            else:
                meta_df              = estimate_smcse(current_df)
            meta_df['bootstrap'] = n_bootstraps
            meta_df['backend']   = zoo_model
            meta_df['dataset']   = current_dataset
            meta_df['estimator'] = estimator
            meta_df['estimand']  = estimand
            meta_dfs.append(meta_df)
    
    meta_df = pd.concat(meta_dfs)
    meta_df.to_csv(out_metadata_path, sep='\t', index=False)

def batch_loop(
    datasets:str,
    models:str,
    root_dir:str,
    batch_size:int,
    n_epochs:int,
    n_workers:int,
    max_sample_size:int,
    log_scale:int,
    min_sample_size:int,
    absolute_scale:bool,
    n_bootstraps:int,
    initial_weights_dir:str, 
    out_data_path:str,
    out_prediction_path:str,
    current_bootstrap:int=None,
    state_dict_dir:str=None,
    out_metadata_path:str=None, 
    start_seed:str="42",
    shuffle:bool=False,
    num_workers:int=4,
    zoo_models:str=None,
    metric_type:str="AUC",
    frequency:bool=False,
    stratified:bool=False,
    experiment:str=None,
    sampler_mode:str=None,
    input_dim:int=None,
    hidden_size:int=None):

    print("Current start seed", start_seed)
    start_seed = int(start_seed)

    print("Running current bootstrap ", current_bootstrap)
    # initialize datasets
    dataset_list = datasets.split(",")
    print(f"Testing datasets... {dataset_list}")

    # initialize models
    models = models.split(",") 
    print(f"Testing models... {models}")
    estimator, estimand = models
    estimator, initial_estimator_weights_path = model_helper(model=estimator, input_dim=input_dim, hidden_size=hidden_size, initial_weights_dir=initial_weights_dir)
    estimand, initial_estimand_weights_path = model_helper(model=estimand, input_dim=input_dim, hidden_size=hidden_size, initial_weights_dir=initial_weights_dir)

    # initialize transfer learning zoo. 
    using_pretrained = (zoo_models != None)
    if(using_pretrained): 
        zoo_model_list = zoo_models.split(",")
        print(f"Using transfer learning base... {zoo_model_list}")
        torch.hub.set_dir(initial_weights_dir)
    else: 
        print("No transfer learning base.")
        zoo_model_list = [None]


    dfs = list()
    preds_dfs = list()
    # initialize dataset dictionary
    for current_dataset in dataset_list:
        dataset_dict = experiment_helper(experiment=experiment, dataset=current_dataset, root_dir=root_dir, input_dim=input_dim, tl_transforms=using_pretrained, start_seed=start_seed)
        print(dataset_dict)
        #dataset_dict = sampling_helper(dataset=current_dataset, root_dir=root_dir)

        # match training and testing
        train_test_match = dataset_dict['train_test_match']

        for trainset, testset in train_test_match:
            print(f"Running trainset - {trainset}, testset - {testset}")
            sample_sizes = generate_sample_sizes(
                max_sample_size=max_sample_size, 
                log_scale=log_scale, 
                min_sample_size=min_sample_size, 
                absolute_scale=absolute_scale
            )
            print(f"Running sample sizes... {sample_sizes}")
            for zoo_model in zoo_model_list:
                print(f"Using transfer learning model... {zoo_model}")
                for sample_size in sample_sizes:
                    print("Running sample size... ", sample_size)
                    #---- estimation block
                    estimators = get_estimators(
                        model = estimator,
                        training_data = dataset_dict[trainset],
                        sample_size = sample_size,
                        batch_size = batch_size,
                        bootstraps = n_bootstraps,
                        start_seed = start_seed,
                        shuffle = shuffle,
                        initial_weights=initial_estimator_weights_path,
                        num_workers=num_workers, 
                        zoo_model=zoo_model,
                        frequency=frequency,
                        stratified=stratified,
                        current_bootstrap=current_bootstrap,
                        sampler_mode=sampler_mode,
                        input_size=input_dim,
                        hidden_size=hidden_size
                    )
                    

                    estimands, preds_df = get_estimands(
                        model = estimand,
                        input_size = input_dim,
                        training_data = dataset_dict[trainset],
                        validation_data = dataset_dict[testset],
                        sample_size = sample_size,
                        batch_size=batch_size,
                        bootstraps=n_bootstraps,
                        start_seed=start_seed,
                        shuffle=shuffle,
                        metric_type=metric_type,
                        initial_weights=initial_estimand_weights_path,
                        num_workers=num_workers,
                        zoo_model=zoo_model,
                        current_bootstrap=current_bootstrap,
                        sampler_mode=sampler_mode,
                        hidden_size=hidden_size, 
                        out_prediction_path=out_prediction_path
                    )

                    #--- Logging block. 
                    print("Logging results... ")
                    df_dict = {}
                    if(frequency == False):
                        df_dict['estimators']   = estimators

                    if(metric_type == "AUC"):
                        df_dict['estimands']    = estimands
                    df_dict['bootstrap']    = [i for i in range(n_bootstraps)]
                    df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
                    df_dict['backend']      = [f'{str(zoo_model)}' for i in range(n_bootstraps)]
                    df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
                    df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
                    df_dict['trainset']     = [f'{trainset}' for i in range(n_bootstraps)]
                    df_dict['testset']      = [f'{testset}' for i in range(n_bootstraps)]
                    df_dict['dataset']      = [f'{trainset}_{testset}_{current_dataset}' for i in range(n_bootstraps)]

                    preds_df['sample_size'] = sample_size
                    preds_df['backend'] = str(zoo_model)
                    preds_df['estimator'] = f'{estimator}'
                    preds_df['estimand'] = f'{estimand}'
                    preds_df['trainset'] = f'{trainset}'
                    preds_df['testset'] = f'{testset}'
                    preds_df['dataset'] = f'{trainset}_{testset}_{current_dataset}'

                    df = pd.DataFrame(df_dict)
                    print(df)
                    print(estimands)
                    print(estimators)
                    if(metric_type == "sAUC"):
                        if(frequency):
                            outputs = estimators.merge(estimands, on=['label', 'bootstrap'])
                            print(outputs)
                            # outputs = pd.concat([estimands, estimators], axis=1)
                            df_merged = outputs.merge(df, on=['bootstrap', 'sample_size'])
                            # df_merged = pd.concat([df, outputs], axis=1)
                        else:
                            df_merged = pd.concat([df, estimators], axis=1)

                    dfs.append(df_merged)
                    preds_dfs.append(preds_df)

                    gc.collect()
    df = pd.concat(dfs)
    df.to_csv(out_data_path, sep="\t", index=False)

    preds_df = pd.concat(preds_dfs)
    preds_df.to_csv(out_prediction_path, sep="\t", index=False)
    

    ## Evaluating Models.
    train_test_data_list = list(df['dataset'].unique())
    meta_dfs = list() 
    for current_dataset in train_test_data_list:
        for zoo_model in zoo_model_list:
            current_df = df[
                (df['backend'] == str(zoo_model)) &
                (df['dataset'] == str(current_dataset))
            ]
            if(metric_type == "AUC"):
                meta_df              = estimate_mcse(current_df)
            else:
                meta_df              = estimate_smcse(current_df)
            meta_df['bootstrap'] = n_bootstraps
            meta_df['backend']   = zoo_model
            meta_df['dataset']   = current_dataset
            meta_df['estimator'] = estimator
            meta_df['estimand']  = estimand
            meta_dfs.append(meta_df)
    
    meta_df = pd.concat(meta_dfs)
    meta_df.to_csv(out_metadata_path, sep='\t', index=False)


def batch_loop(
    datasets:str,
    models:str,
    root_dir:str,
    batch_size:int,
    n_epochs:int,
    n_workers:int,
    max_sample_size:int,
    log_scale:int,
    min_sample_size:int,
    absolute_scale:bool,
    n_bootstraps:int,
    initial_weights_dir:str, 
    out_data_path:str,
    out_prediction_path:str,
    current_bootstrap:int=None,
    state_dict_dir:str=None,
    out_metadata_path:str=None, 
    start_seed:str="42",
    shuffle:bool=False,
    num_workers:int=4,
    zoo_models:str=None,
    metric_type:str="AUC",
    frequency:bool=False,
    stratified:bool=False,
    experiment:str=None,
    sampler_mode:str=None,
    input_dim:int=None,
    hidden_size:int=None):

    print("Current start seed", start_seed)
    start_seed = int(start_seed)

    print("Running current bootstrap ", current_bootstrap)
    # initialize datasets
    dataset_list = datasets.split(",")
    print(f"Testing datasets... {dataset_list}")

    # initialize models
    models = models.split(",") 
    print(f"Testing models... {models}")
    estimator, estimand = models
    estimator, initial_estimator_weights_path = model_helper(model=estimator, input_dim=input_dim, hidden_size=hidden_size, initial_weights_dir=initial_weights_dir)
    estimand, initial_estimand_weights_path = model_helper(model=estimand, input_dim=input_dim, hidden_size=hidden_size, initial_weights_dir=initial_weights_dir)

    # initialize transfer learning zoo. 
    using_pretrained = (zoo_models != None)
    if(using_pretrained): 
        zoo_model_list = zoo_models.split(",")
        print(f"Using transfer learning base... {zoo_model_list}")
        torch.hub.set_dir(initial_weights_dir)
    else: 
        print("No transfer learning base.")
        zoo_model_list = [None]


    dfs = list()
    preds_dfs = list()
    # initialize dataset dictionary
    for current_dataset in dataset_list:
        dataset_dict = experiment_helper(experiment=experiment, dataset=current_dataset, root_dir=root_dir, input_dim=input_dim, tl_transforms=using_pretrained, start_seed=start_seed)
        print(dataset_dict)
        #dataset_dict = sampling_helper(dataset=current_dataset, root_dir=root_dir)

        # match training and testing
        train_test_match = dataset_dict['train_test_match']

        for trainset, testset in train_test_match:
            print(f"Running trainset - {trainset}, testset - {testset}")
            sample_sizes = generate_sample_sizes(
                max_sample_size=max_sample_size, 
                log_scale=log_scale, 
                min_sample_size=min_sample_size, 
                absolute_scale=absolute_scale
            )
            print(f"Running sample sizes... {sample_sizes}")
            for zoo_model in zoo_model_list:
                print(f"Using transfer learning model... {zoo_model}")
                for sample_size in sample_sizes:

                    #---- estimation block
                    estimators = get_estimators(
                        model = estimator,
                        training_data = dataset_dict[trainset],
                        sample_size = sample_size,
                        batch_size = batch_size,
                        bootstraps = n_bootstraps,
                        start_seed = start_seed,
                        shuffle = shuffle,
                        initial_weights=initial_estimator_weights_path,
                        num_workers=num_workers, 
                        zoo_model=zoo_model,
                        frequency=frequency,
                        stratified=stratified,
                        current_bootstrap=current_bootstrap,
                        sampler_mode=sampler_mode,
                        input_size=input_dim,
                        hidden_size=hidden_size
                    )
                    

                    estimands, preds_df = get_estimands(
                        model = estimand,
                        input_size = input_dim,
                        training_data = dataset_dict[trainset],
                        validation_data = dataset_dict[testset],
                        sample_size = sample_size,
                        batch_size=batch_size,
                        bootstraps=n_bootstraps,
                        start_seed=start_seed,
                        shuffle=shuffle,
                        metric_type=metric_type,
                        initial_weights=initial_estimand_weights_path,
                        num_workers=num_workers,
                        zoo_model=zoo_model,
                        current_bootstrap=current_bootstrap,
                        sampler_mode=sampler_mode,
                        hidden_size=hidden_size, 
                        out_prediction_path=out_prediction_path
                    )

                    #--- Logging block. 
                    print("Logging results... ")
                    df_dict = {}
                    if(frequency == False):
                        df_dict['estimators']   = estimators

                    if(metric_type == "AUC"):
                        df_dict['estimands']    = estimands
                    df_dict['bootstrap']    = [i for i in range(n_bootstraps)]
                    df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
                    df_dict['backend']      = [f'{str(zoo_model)}' for i in range(n_bootstraps)]
                    df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
                    df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
                    df_dict['trainset']     = [f'{trainset}' for i in range(n_bootstraps)]
                    df_dict['testset']      = [f'{testset}' for i in range(n_bootstraps)]
                    df_dict['dataset']      = [f'{trainset}_{testset}_{current_dataset}' for i in range(n_bootstraps)]

                    preds_df['sample_size'] = sample_size
                    preds_df['backend'] = str(zoo_model)
                    preds_df['estimator'] = f'{estimator}'
                    preds_df['estimand'] = f'{estimand}'
                    preds_df['trainset'] = f'{trainset}'
                    preds_df['testset'] = f'{testset}'
                    preds_df['dataset'] = f'{trainset}_{testset}_{current_dataset}'

                    df = pd.DataFrame(df_dict)
                    print(df)
                    print(estimands)
                    print(estimators)
                    if(metric_type == "sAUC"):
                        if(frequency):
                            outputs = estimators.merge(estimands, on=['label', 'bootstrap'])
                            print(outputs)
                            # outputs = pd.concat([estimands, estimators], axis=1)
                            df_merged = outputs.merge(df, on=['bootstrap', 'sample_size'])
                            # df_merged = pd.concat([df, outputs], axis=1)
                        else:
                            df_merged = pd.concat([df, estimators], axis=1)

                    dfs.append(df_merged)
                    preds_dfs.append(preds_df)

                    gc.collect()
    df = pd.concat(dfs)
    df.to_csv(out_data_path, sep="\t", index=False)

    preds_df = pd.concat(preds_dfs)
    preds_df.to_csv(out_prediction_path, sep="\t", index=False)
    

    ## Evaluating Models.
    train_test_data_list = list(df['dataset'].unique())
    meta_dfs = list() 
    for current_dataset in train_test_data_list:
        for zoo_model in zoo_model_list:
            current_df = df[
                (df['backend'] == str(zoo_model)) &
                (df['dataset'] == str(current_dataset))
            ]
            if(metric_type == "AUC"):
                meta_df              = estimate_mcse(current_df)
            else:
                meta_df              = estimate_smcse(current_df)
            meta_df['bootstrap'] = n_bootstraps
            meta_df['backend']   = zoo_model
            meta_df['dataset']   = current_dataset
            meta_df['estimator'] = estimator
            meta_df['estimand']  = estimand
            meta_dfs.append(meta_df)
    
    meta_df = pd.concat(meta_dfs)
    meta_df.to_csv(out_metadata_path, sep='\t', index=False)

def measure_disparity_custom(
    dataset_path:str ,
    demographics_cols:str,
    outcome_cols:str,
    exclude_cols:str,
    root_dir:str,
    models:str,
    input_dim:int=None,
    hidden_size:int=None,
    n_bootstraps:int=50,
    zoo_models:str=None,
    max_sample_size:int=5000,
    log_scale=2, 
    min_sample_size=16, 
    absolute_scale=False,
    sampler_mode=None,
    config:str=None,
    start_seed:int=42,
    split_ratio:float=0.8,
    batch_size:int=1,
    shuffle:bool=True,
    num_workers:int=0,
    frequency:bool=True,
    stratified:bool=True,
    ):

    # Seeding experiments. 
    start_seed = int(start_seed)

    # Initialize datasets. 
    dataset_dict = custom_helper(
        dataset_path=dataset_path, 
        root_dir=root_dir,
        start_seed=start_seed,
        outcome_cols=outcome_cols,
        demographics_col=demographics_cols,
        exclude_cols=exclude_cols,
        split_ratio=split_ratio
    )
    
    # initialize models
    models = models.split(",") 
    #print(f"Testing models... {models}")
    estimator, estimand = models
    estimator, initial_estimator_weights_path = model_helper(
        model=estimator, 
        input_dim=input_dim, hidden_size=hidden_size, 
        initial_weights_dir=root_dir)
    estimand, initial_estimand_weights_path = model_helper(
        model=estimand, input_dim=input_dim, hidden_size=hidden_size, 
        initial_weights_dir=root_dir)

    
        
    # Get unique training datasets. 
    train_test_match = dataset_dict['train_test_match']
    trainsets = [datasets[0] for datasets in train_test_match]
    trainsets = [*set(trainsets)]
    #print("Running trainsets", trainsets)

    # Calculate sample sizes for each sample size. 
    sample_sizes = generate_sample_sizes(
            max_sample_size=max_sample_size, 
            log_scale=log_scale, 
            min_sample_size=min_sample_size, 
            absolute_scale=absolute_scale
    )

    # Calculating AEq values for each dataset in the training dataset list. 
    dfs = list()
    for trainset in trainsets:

        #---- Initialize sample size. 
        for sample_size in sample_sizes:

            #---- estimation block
            estimators = get_estimators(
                model = estimator,
                training_data = dataset_dict[trainset],
                sample_size = sample_size,
                batch_size = batch_size,
                bootstraps = n_bootstraps,
                start_seed = start_seed,
                shuffle = shuffle,
                initial_weights=initial_estimator_weights_path,
                num_workers=num_workers, 
                zoo_model=None,
                frequency=frequency,
                stratified=stratified,
                current_bootstrap=None,
                sampler_mode=sampler_mode,
                input_size=input_dim,
                hidden_size=hidden_size
            )

            #--- Logging block. 
            #print(estimators)

            df_dict = {}
            #df_dict['estimators']   = estimators
            #df_dict['bootstrap']    = [i for i in range(n_bootstraps)]
            #df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
            df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
            df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
            df_dict['dataset']     = [f'{trainset}' for i in range(n_bootstraps)]
            df_dict['sample_size'] = [sample_size for i in range(n_bootstraps)]
            #print(df_dict)

            df = pd.DataFrame(df_dict)
            df_merged = df.merge(estimators, on = 'sample_size')

            dfs.append(df_merged)

            gc.collect()

    df = pd.concat(dfs)
    df.reset_index(inplace=True)
    out_data_path = os.path.join(root_dir, 'data.tsv')
    df.to_csv(out_data_path, sep="\t", index=False)


    ## Calculate AEq.
    datasets = list(df["dataset"].drop_duplicates())
    labels = list(df["label"].drop_duplicates())
    bootstraps = list(df["bootstrap"].drop_duplicates())

    datasets_vector = []
    labels_vector = []
    bootstraps_vector = []
    mcse_vector = []

    for value, item, version in itertools.product(labels, datasets, bootstraps):
        datasets_vector.append(item)
        labels_vector.append(value)
        bootstraps_vector.append(version)
        mcse_vector.append(get_mcse_discrete(df, value, item, int(version)))

    metadata = pd.DataFrame({
        "dataset": datasets_vector,
        "label": labels_vector,
        "bootstrap": bootstraps_vector,
        "mcse": mcse_vector
    })

    vc = metadata.groupby(["dataset", "label"]).agg({
        "mcse": ["mean", "std"]
    })
    vc["aeq_mean"] = vc["mcse"]["mean"]
    vc["aeq_sd"] = vc["mcse"]["std"] / len(bootstraps_vector)
    vc.drop(["mcse"], axis=1, inplace=True)
    vc.reset_index(inplace=True)

    # Initialize file paths
    out_data_path = os.path.join(root_dir, 'data.tsv')
    out_metadata_path = os.path.join(root_dir, 'metadata.tsv')

    df.to_csv(out_data_path, sep="\t", index=False)
    vc.to_csv(out_metadata_path, sep='\t', index=False)

    # Reformat with appropriate headers. 
    vc = pd.read_csv(out_metadata_path, sep="\t")
    vc.dropna(subset=['dataset'], inplace=True)
    vc[['demographics', 'outcome', 'subset']] = vc['dataset'].str.split("__", expand=True)
    vc.to_csv(out_metadata_path, sep='\t', index=False)


def mitigate_disparity_custom(
    dataset_path:str ,
    demographics_cols:str,
    outcome_cols:str,
    exclude_cols:str,
    root_dir:str,
    models:str,
    input_dim:int=None,
    hidden_size:int=None,
    start_seed:int=42,
    zoo_models:str=None,
    max_sample_size:int=5000,
    sampler_mode=None,
    frequency:bool=True,
    stratified:bool=True,
    n_bootstraps:int=50,
    n_classes:int=2, 
    config:str=None,
    log_scale=2, 
    min_sample_size=16, 
    absolute_scale=False,
    batch_size:int=1,
    shuffle:bool=False,
    num_workers:int=4,
    split_ratio:float=0.8,
    current_dataset:str="custom",
    metric_type:str="sAUC"
    ):

    # Seeding experiments. 
    start_seed = int(start_seed)

    # Initialize file paths
    out_prediction_path = os.path.join(root_dir, 'predictions.tsv')
    out_data_path = os.path.join(root_dir, 'mitigated.tsv')

    # Initialize datasets. 
    dataset_dict = custom_helper(
        dataset_path=dataset_path, 
        root_dir=root_dir,
        start_seed=start_seed,
        outcome_cols=outcome_cols,
        demographics_col=demographics_cols,
        exclude_cols=exclude_cols,
        split_ratio=split_ratio
    )
    
    # Initialize tested sample sizes. 
    sample_sizes = generate_sample_sizes(
            max_sample_size=max_sample_size, 
            log_scale=log_scale, 
            min_sample_size=min_sample_size, 
            absolute_scale=absolute_scale
    )

    # Initialize models
    models = models.split(",") 
    estimator, estimand = models
    estimator, initial_estimator_weights_path = model_helper(
        model=estimator, 
        input_dim=input_dim, hidden_size=hidden_size, output_size=n_classes,
        initial_weights_dir=root_dir
    )
    estimand, initial_estimand_weights_path = model_helper(
        model=estimand, input_dim=input_dim, hidden_size=hidden_size, output_size=n_classes,
        initial_weights_dir=root_dir)

    # Get unique training datasets. 
    train_test_match = dataset_dict['train_test_match']

    # Iterate over the training dictionary. 
    dfs = list()
    preds_dfs = list()
    for trainset, testset in train_test_match:
        
        #---- Initialize sample size. 
        for sample_size in sample_sizes:
            
            estimands, preds_df = get_estimands(
                model = estimand,
                input_size = input_dim,
                training_data = dataset_dict[trainset],
                validation_data = dataset_dict[testset],
                sample_size = sample_size,
                batch_size=batch_size,
                bootstraps=n_bootstraps,
                start_seed=start_seed,
                shuffle=shuffle,
                metric_type=metric_type,
                initial_weights=initial_estimand_weights_path,
                num_workers=num_workers,
                zoo_model=None,
                current_bootstrap=None,
                sampler_mode=sampler_mode,
                hidden_size=hidden_size, 
                output_size=n_classes,
                out_prediction_path=out_prediction_path
            )

            #--- Logging block. 
            # print("Logging results... ")
            # df_dict = {}

            # if(metric_type == "AUC"):
            # df_dict['estimands']    = estimands
            estimands['sample_size']    = sample_size
            estimands['trainset']       = f'{trainset}'
            estimands['testset']        = f'{testset}'
            # estimands['bootstrap'] 
            # df_dict['bootstrap']    = [i for i in range(n_bootstraps)]
            # df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
            # #df_dict['backend']      = [f'{str(zoo_model)}' for i in range(n_bootstraps)]
            # df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
            # df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
            # df_dict['trainset']     = [f'{trainset}' for i in range(n_bootstraps)]
            # df_dict['testset']      = [f'{testset}' for i in range(n_bootstraps)]
            # df_dict['dataset']      = [f'{trainset}_{testset}_{current_dataset}' for i in range(n_bootstraps)]

            preds_df['sample_size'] = sample_size
            preds_df['estimator'] = f'{estimator}'
            preds_df['estimand'] = f'{estimand}'
            preds_df['trainset'] = f'{trainset}'
            preds_df['testset'] = f'{testset}'
            preds_df['dataset'] = f'{trainset}_{testset}_{current_dataset}'

            # Collect data frames. 
            dfs.append(estimands)
            preds_dfs.append(preds_df)

            gc.collect()
    
    
    # Postprocess dataframe
    df = pd.concat(dfs)
    df[['demographics_train',   'outcome_train', 'subset_train']] = df['trainset'].str.split("__", expand=True)
    df[['demographics_test',    'outcome_test',  'subset_test']]  = df['testset'].str.split("__", expand=True)
    df.to_csv(out_data_path, sep="\t", index=False)
    
    
    preds_df = pd.concat(preds_dfs)
    preds_df[['demographics_train',   'outcome_train', 'subset_train']] = preds_df['trainset'].str.split("__", expand=True)
    preds_df[['demographics_test',    'outcome_test',  'subset_test']]  = preds_df['testset'].str.split("__", expand=True)
    preds_df.to_csv(out_prediction_path, sep="\t", index=False)

    return df, preds_df