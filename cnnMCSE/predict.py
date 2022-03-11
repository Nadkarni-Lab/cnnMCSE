"""Generate estimators and estimands for sample size estimation with convolutional neural networks. 
"""
import gc
import torch
from pyexpat import model
import pandas as pd

from cnnMCSE.dataloader import dataloader_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.helpers import generate_sample_sizes, get_derivative, get_inflection_point
from cnnMCSE.mcse import get_estimators, get_estimands
from cnnMCSE.utils.helpers import estimate_mcse

def predict_loop(
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
    zoo_models:str=None
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
        print("Using transfer learning base... {zoo_model_list}")
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
                    zoo_model=zoo_model
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
                    metric_type="AUC",
                    initial_weights=initial_estimand_weights_path,
                    num_workers=num_workers,
                    zoo_model=zoo_model
                )

                #--- Logging block. 
                print("Logging results... ")
                df_dict = {}
                df_dict['estimators']   = estimators
                df_dict['estimands']    = estimands
                df_dict['bootstrap']    = [i+1 for i in range(n_bootstraps)]
                df_dict['sample_size']  = [sample_size for i in range(n_bootstraps)]   
                df_dict['backend']      = [f'{str(zoo_model)}' for i in range(n_bootstraps)]
                df_dict['estimator']    = [f'{estimator}' for i in range(n_bootstraps)] 
                df_dict['estimand']     = [f'{estimand}' for i in range(n_bootstraps)] 
                df_dict['dataset']      = [f'{current_dataset}' for i in range(n_bootstraps)]
                df = pd.DataFrame(df_dict)
                dfs.append(df)

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
            meta_df              = estimate_mcse(current_df, out_metadata_path)
            meta_df['bootstrap'] = n_bootstraps
            meta_df['backend']   = zoo_model
            meta_df['dataset']   = current_dataset
            meta_df['estimator'] = estimator
            meta_df['estimand']  = estimand
            meta_dfs.append(meta_df)
    
    meta_df = pd.concat(meta_dfs)
    meta_df.to_csv(out_metadata_path, sep='\t', index=False)




    # print(sample_sizes)
    # # sample_sizes = sample_sizes[0:2]
    # dfs = list()
    # for zoo_model in zoo_model_list:
    #     for sample_size in sample_sizes:
    #         df_dict = {}
    #         print(sample_size)
    #         estimators = get_estimators(
    #             model = estimator,
    #             training_data = trainset,
    #             sample_size = sample_size,
    #             batch_size = batch_size,
    #             bootstraps = n_bootstraps,
    #             start_seed = start_seed,
    #             shuffle = shuffle,
    #             initial_weights=initial_estimator_weights_path,
    #             num_workers=num_workers, 
    #             zoo_model=zoo_model
    #         )
            

    #         estimands = get_estimands(
    #             model = estimand,
    #             training_data = trainset,
    #             validation_data = testset,
    #             sample_size = sample_size,
    #             batch_size=batch_size,
    #             bootstraps=n_bootstraps,
    #             start_seed=start_seed,
    #             shuffle=shuffle,
    #             metric_type="AUC",
    #             initial_weights=initial_estimand_weights_path,
    #             num_workers=num_workers,
    #             zoo_model=zoo_model
    #         )

    #         df_dict['estimators'] = estimators
    #         df_dict['estimands']  = estimands
    #         df_dict['bootstrap']  = [i+1 for i in range(n_bootstraps)]
    #         df_dict['sample_size'] = [sample_size for i in range(n_bootstraps)]     
    #         df_dict['model'] = [f'{model}_{str(zoo_model)}' for i in range(n_bootstraps)] 
    #         df = pd.DataFrame(df_dict)
    #         dfs.append(df)
        
    #         print(estimators)
    #         print(estimands)
    #         gc.collect()
    
    # df = pd.concat(dfs)
    # df.to_csv(out_data_path, sep="\t")
    #print("Estimating MCSE")
    #estimate_mcse(df, out_metadata_path)

    print("Complete")












# def msse_predict(
#     max_sample_size : int = 1000, 
#     log_scale: int = 2, 
#     min_sample_size: int = 64, 
#     absolute_scale = None,
#     n_informative = 78, 
#     n_features = 784, 
#     n_classes=10,
#     bootleg=2,
#     hidden_size_one: int = 1024, 
#     hidden_size_two : int = 512, 
#     hidden_size_three: int = 256, 
#     latent_classes: int = 2):
    
#     # generate sample sizes. 
#     sample_sizes = generate_sample_sizes(max_sample_size=max_sample_size, log_scale=log_scale, min_sample_size=min_sample_size, absolute_scale=absolute_scale)
    
#     # generate dataset. 
#     sample_dataset = make_classification(n_samples=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes)
#     tensor_x = torch.Tensor(sample_dataset[0]) 
#     tensor_y = torch.LongTensor(sample_dataset[1]) 
#     sample_dataset = TensorDataset(tensor_x,tensor_y) 
    
#     # collect bootstraps. 
#     A3_losses = list()
#     for sample_size in sample_sizes:
#         model_current_sample_size, training_loss_current_sample_size = get_models_bootleg_A3(
#                             sample_size=sample_size,
#                             trainset=sample_dataset,
#                             bootleg=bootleg,
#                             input_size=n_features,
#                             hidden_size_one = hidden_size_one, 
#                             hidden_size_two = hidden_size_two, 
#                             hidden_size_three = hidden_size_three, 
#                             latent_classes = latent_classes)
#         A3_losses.append(training_loss_current_sample_size)


#     print(A3_losses)
#     print(sample_sizes)
#     A3_loss_final = list()
#     sample_size_final = list()

#     for sample_size, A3_loss in zip(sample_sizes, A3_losses):
#         A3_loss_2 = [x for x in A3_loss if np.isnan(x) == False]
#         if(len(A3_loss_2) != 0):
#             A3_loss_final.append(A3_loss_2)
#             sample_size_final.append(sample_size)

#     print(A3_loss_final)
#     print(sample_size_final)

#     y_spl, y_spl_1D, y_spl_2D         = get_derivative(A3_loss_raw=A3_loss_final, sample_sizes=sample_size_final)
#     y_infl, y_infl_erd, y_infl_eru = get_inflection_point(y_spl, y_spl_2D, sample_sizes=sample_size_final)

#     msse_estimate = np.exp(y_infl)
#     print(msse_estimate)
#     return msse_estimate
    # pass