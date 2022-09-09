"""File to help with experimentation.
"""
from cnnMCSE.dataloader import synthetic_dataset
def informative_synthetic(max_sample_size:int = 6000, n_informative_list:list=[2,4,8,16,32,64,128,256,512], n_features:int=784, n_classes:int=10): 
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    n_informative_list=[20, 40, 60, 80, 100, 150, 200, 400, 600]
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

def classes_synthetic(max_sample_size:int = 6000, n_informative:int=128, n_features:int=784, n_classes_list:list=[2, 4, 8, 10, 15, 20, 40]): 
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    n_classes_list=[2, 4, 8, 10, 15, 20, 40]
    print(n_classes_list)
    for n_classes in n_classes_list:
        subsample_name = f'{n_classes}'
        print(max_sample_size)
        print(n_classes)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

        dataset_dict[f'{n_classes}_trainset'] = trainset
        dataset_dict[f'{n_classes}_testset'] = testset
        dataset_dict['train_test_match'].append(
            [f'{n_classes}_trainset',  f'{n_classes}_testset']
        )
    
    return dataset_dict

def features_synthetic(max_sample_size:int = 6000, n_informative:int=128, n_features:int=784, n_classes_list:list=[2, 4, 8, 10, 15, 20, 40]): 
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    n_classes_list=[2, 4, 8, 10, 15, 20, 40]
    print(n_classes_list)
    for n_classes in n_classes_list:
        subsample_name = f'{n_dim}'
        print(max_sample_size)
        print(n_dim)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

        dataset_dict[f'{n_dim}_trainset'] = trainset
        dataset_dict[f'{n_dim}_testset'] = testset
        dataset_dict['train_test_match'].append(
            [f'{n_classes}_trainset',  f'{n_classes}_testset']
        )
    
    return dataset_dict


def flip_synthetic(max_sample_size:int = 6000, n_informative:int=128, n_features:int=784, n_classes:int=10): 
    dataset_dict = {}
    dataset_dict['train_test_match'] = list()
    flip_y_list = [0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    for flip_y in flip_y_list:
        subsample_name = f'{n_dim}'
        print(max_sample_size)
        print(n_dim)
        print(n_features)
        print(n_classes)
        trainset, testset = synthetic_dataset(max_sample_size=max_sample_size, n_informative=n_informative, n_features=n_features, n_classes=n_classes, train_test_split=5000, seed=42)

        dataset_dict[f'{n_dim}_trainset'] = trainset
        dataset_dict[f'{n_dim}_testset'] = testset
        dataset_dict['train_test_match'].append(
            [f'{n_classes}_trainset',  f'{n_classes}_testset']
        )
    
    return dataset_dict





def synthetic_helper(root_dir, dataset, tl_transforms:bool=False):
    if(dataset == "informative"):
        return informative_synthetic()
    if(dataset == "classes"):
        return classes_synthetic()
    if(dataset == "features"):
        return classes_synthetic()
    else:
        return None

