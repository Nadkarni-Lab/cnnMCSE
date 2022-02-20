from cnnMCSE.dataloader import dataloader_helper
from cnnMCSE.models import model_helper
from cnnMCSE.utils.helpers import generate_sample_sizes, get_derivative, get_inflection_point

def predict_loop(
    dataset:str,
    models:str,
    root_dir:str,
    batch_size:int,
    n_epochs:int,
    n_workers:int,
    max_sample_size:int,
    log_scale:int,
    min_sample_size:int,
    absolute_scale:bool
    ):

    # return the training and testing datasets
    trainset, testset = dataloader_helper(dataset, root_dir)

    # return the models.
    models = models.split(",") 
    estimator, estimand = models
    estimator = model_helper(estimator)
    estimand = model_helper(estimand)

    sample_sizes = generate_sample_sizes(
        max_sample_size=max_sample_size, 
        log_scale=log_scale, 
        min_sample_size=min_sample_size, 
        absolute_scale=absolute_scale
    )

    print(sample_sizes)



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