import os
import pandas as pd
import torch

working_dir = '/sc/arion/projects/mscic1/data/nih-cxr'
data_entry = os.path.join(working_dir, 'Data_Entry_2017.csv')
bbox = os.path.join(working_dir, 'BBox_List_2017.csv')

def label_encoding(labels_df:pd.DataFrame, all_labels:list, include_opacity:bool=True)->pd.DataFrame:
    """_summary_

    Args:
        labels_df (pd.DataFrame): DataFrame where to generate labels
        all_labels (list): List of relevant labels. 

    Returns:
        pd.DataFrame: Dataframe with encoded labels. 
    """
    # Custom selected labels to represent opacity. 
    opacity_list = [
        'Infiltration',
        'Mass',
        'Nodule',
        'Fibrosis'
    ]

    target = list()
    target_map = list()
    for index, row in data_df.iterrows():
        current_target = len(all_labels) + 2
        current_label = "Other"
        labels_list = row['split_labels']
        for val, label in enumerate(all_labels):
            if(label in labels_list):
                current_target = val
                current_label = label
        if(current_label == "Other" and include_opacity):
            for val, label in enumerate(opacity_list):
                if(label in labels_list):
                    current_target = len(all_labels)
                    current_label = "Lung Opacity"
        target.append(current_target)
        target_map.append(current_label)
    data_df['target'] = target
    data_df['target_map'] = target_map


def generate_metadata_file(root_dir:str, labels:str, genders:str=None, age_range:str=None, orientations:str=None):

    # Initialize out config file. 
    out_config = ["NIH"]
    labels = labels.split(",")
    out_config = out_config + labels

    if(orientations):
        orientations = orientations.split(",")
        out_config = out_config + orientations

    if(genders): 
        genders = genders.split(",")
        out_config = out_config + genders
    if(age_range): 
        age_range = age_range.split(",")
        out_config = out_config + age_range

    data_df = pd.read_csv(data_entry)
    bbox_df = pd.read_csv(bbox)
    

    data_df = label_encoding(
        labels_df=data_df, all_labels=all_labels, include_opacity=include_opacity
    )

    # Add genders. 
    if(genders):
        data_df = data_df[
            data_df['Patient Gender'].isin(genders)
        ]
    
    # Add age range. 
    if(age_range):
        age_range = [int(age) for age in age_range]
        data_df = data_df[
            data_df['Patient Age'].between(age_range*)
        ]
    
    # Add file paths.
    image_dirs = os.listdir(working_dir)
    image_dirs = [image_dir for image_dir in image_dirs if 'images_' in image_dir]
    image_dirs = [os.path.join(working_dir, image_dir) for image_dir in image_dirs]
    image_dirs = [os.path.join(image_dir, 'images') for image_dir in image_dirs]

    # Image Paths. 
    image_paths = list()
    current_image_all = list()
    for image_dir in image_dirs:
        current_images = os.listdir(image_dir)
        current_image_all = current_image_all + current_images
        current_images = [os.path.join(image_dir, current_image) for current_image in current_images]
        image_paths = image_paths + current_images
    path_df = pd.DataFrame({
        'Image Index': current_image_all,
        'path' : image_paths
    })

    # Output file. 
    output_df = pd.merge(data_df, path_df, on = "Image Index")
    out_config = "_".join(out_config)
    output_path = os.path.join(root_dir, out_config) + ".tsv"
    output_df.to_csv(output_path, sep="\t", index=False)

    return output_path

def generate_dataloaders(
        metadata_path:str=None, 
        metadata_paths:str=None,
        tl_transforms:bool=False,
        split_ratio:float=0.8, 
        start_seed:int=42
    ):
    print('TL transforms', tl_transforms)
    generator_1 = torch.Generator().manual_seed(start_seed)
    generator_2 = torch.Generator().manual_seed(start_seed+1)
    if(metadata_paths):
        metadata_path_1, metadata_path_2 = metadata_paths
    else:
        metadata_path_1 = metadata_path
        metadata_path_2 = metadata_path
    # generator_2 = torch.Generator().manual_seed(start_seed+1)

    if(tl_transforms == False):
        to_grayscale = transforms.Grayscale()
        resize = transforms.Resize((28,28))
        normalize = transforms.Normalize((0.5,), (0.5,))
        transform = transforms.Compose([
            resize,
            to_grayscale,
            transforms.ToTensor(),
            normalize
        ])
    elif(tl_transforms == True):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
        resize = transforms.Resize((224, 224))
        transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset_1 = NihCXRDataset(
        metadata_path=metadata_path_1,
        transform=transform
    )
    len_dataset_1 = len(dataset_1)
    len_trainset_1 = int(len_dataset_1 * split_ratio)
    len_testset_1 = len_dataset_1 - len_trainset_1
    trainset_1, testset_1 = random_split(dataset_1, [len_trainset_1, len_testset_1], generator=generator_1)

    dataset_2 = NihCXRDataset(
        metadata_path=metadata_path_2,
        transform=transform
    )
    len_dataset_2 = len(dataset_2)
    len_trainset_2 = int(len_dataset_2 * split_ratio)
    len_testset_2 = len_dataset_2 - len_trainset_2
    trainset_2, testset_2 = random_split(dataset_2, [len_trainset_2, len_testset_2], generator=generator_2)
    # trainset_2, testset_2 = random_split(dataset, [len_dataset*split_ratio, len_dataset*(1-split_ratio)], generator=generator_2)
    joint_trainset = ConcatDataset([trainset_1, trainset_2])
    joint_testset = ConcatDataset([testset_1, testset_2])

    dataset_dict = {
        'subsample_1': trainset_1, 
        'testsample_1': testset_1, 
        'subsample_2' : trainset_2,
        'testsample_2': testset_2,
        'joint_trainset' : joint_trainset,
        'joint_testset' : joint_testset,
        'train_test_match': [
            ['subsample_1', 'testsample_1'], 
            ['subsample_2', 'testsample_2'], 
            ['joint_trainset', 'joint_testset'], 
            ['joint_trainset', 'testsample_1'],
            ['joint_trainset', 'testsample_2']
        ]
    }
    return dataset_dict

def nih_helper(dataset, root_dir, tl_transforms:bool=True):
    if(dataset == "test"):
        metadata_path = generate_metadata_file(
            labels='No Finding,Cardiomegaly,Pneumothorax,Effusion,Edema,Consolidation,Atelectasis,Pneumonia,Consolidation',
            include_opacity=True
        )
        return generate_dataloaders(metadata_paths=[metadata_path, metadata_path], tl_transforms=tl_transforms)
    
    if(dataset == "gender"):
        metadata_path_1 = generate_metadata_file(
            labels='No Finding,Cardiomegaly,Pneumothorax,Effusion,Edema,Consolidation,Atelectasis,Pneumonia,Consolidation',
            include_opacity=True,
            gender="F"
        )

        metadata_path_2 = generate_metadata_file(
            labels='No Finding,Cardiomegaly,Pneumothorax,Effusion,Edema,Consolidation,Atelectasis,Pneumonia,Consolidation',
            include_opacity=True,
            gender="M"
        )
        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2], tl_transforms=tl_transforms)
    
    if(dataset == "age"):
        metadata_path_1 = generate_metadata_file(
            labels='No Finding,Cardiomegaly,Pneumothorax,Effusion,Edema,Consolidation,Atelectasis,Pneumonia,Consolidation',
            include_opacity=True,
            age_range="20,40"
        )

        metadata_path_2 = generate_metadata_file(
            labels='No Finding,Cardiomegaly,Pneumothorax,Effusion,Edema,Consolidation,Atelectasis,Pneumonia,Consolidation',
            include_opacity=True,
            age_range="40,60"
        )
        return generate_dataloaders(metadata_paths=[metadata_path_1, metadata_path_2], tl_transforms=tl_transforms)






class NihCXRDataset(Dataset):
    """Dataset for NIH CXR. 
    """

    def __init__(self, metadata_path, transform=None, target_transform=None):
        self.metadata_path = metadata_path
        self.metadata_df = pd.read_csv(metadata_path, sep="\t")
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.metadata_df.shape[0]
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        path = row['path']
        img = Image.open(path)
        target = row['target']

        if(self.transform):
            img = self.transform(img)
        
        if(self.target_transform):
            target = self.target_transform(target)
        
        return img, target

