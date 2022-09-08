"""File to help with experimentation.
"""
from cnnMCSE.dataloader import synthetic_dataset
def informative_synthetic(max_sample_size:int = 6000, n_informative_list:list=[2,4,8,16,32,64,128,256,512], n_features:int=784, n_classes:int=2): 
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    n_informative_list=[4,8,16,32,64,128,256,512]
    print(n_informative_list)
    for n_dim in n_informative_list:
        subsample_name = f'{n_dim}'
        print(max_sample_size)
        print(n_dim)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_dim, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

        dataset_dict[f'{n_dim}_trainset'] = trainset
        dataset_dict[f'{n_dim}_testset'] = testset
        dataset_dict['train_test_match'].append(
            [f'{n_dim}_trainset',  f'{n_dim}_testset']
        )
    
    return dataset_dict




def synthetic_helper(root_dir, dataset, tl_transforms:bool=False):
    if(dataset == "informative"):
        return informative_synthetic()
    else:
        return None

