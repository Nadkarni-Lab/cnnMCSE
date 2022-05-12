"""Generate estimators and estimands for sample size estimation with convolutional neural networks. 
"""
import gc
import torch
from pyexpat import model
import pandas as pd

from cnnMCSE.dataloader import dataloader_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.helpers import experiment_helper, generate_sample_sizes, get_derivative, get_inflection_point
from cnnMCSE.mcse import get_estimators, get_estimands
from cnnMCSE.utils.helpers import estimate_mcse, estimate_smcse, experiment_helper
from cnnMCSE.experiments.sampling import sampling_helper

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
    experiment:str=None
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
        dataset_dict = experiment_helper(experiment=experiment, dataset=current_dataset, root_dir=root_dir, tl_transforms=using_pretrained)
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
                        num_workers=num_workers,
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