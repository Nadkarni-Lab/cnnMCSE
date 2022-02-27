"""Metrics.py contains all the metrics for machine learning validation. 
"""
import torch
import torch.nn as nn

from sklearn import metrics
from sklearn.preprocessing import label_binarize

def get_AUC(model, loader=None, dataset=None, num_workers:int=0, num_classes:int=10):
    """Get Area-Under-Curve Metric on a test dataset. 

    Args:
        model (_type_): Model to validate. 
        dataset (_type_): Dataset to use. 

    Returns:
        float: AUC metric. 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    current_model = model
    current_model = nn.DataParallel(current_model)
    current_model.to(device)
    current_model.eval()

    model_predictions = []
    model_labels      = []

    print("Generating predictions")
    print(len(loader))
    #with torch.no_grad():
    print("Broken here....")
    for index, data in enumerate(loader):
        print("Running index", index)
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # images, labels = images.to(DEVICE), labels.to(DEVICE)
        output = current_model(images)
        _, predicted = torch.max(output.data, 1)
        model_predictions = model_predictions + predicted.tolist()
        model_labels = model_labels + labels.tolist()
            # model_predictions.append(predicted.tolist())
            #print('Predictions', predicted.shape)
        # for prediction in predicted:
        #     #print('Predicted', prediction.shape)
        #     model_predictions.append(prediction.tolist())
        # for label in labels:
        #     #print('Labels', label.shape)
        #     model_labels.append(label.tolist())


    print('Model Labels', len(model_labels))
    print('Model Predictions',len(model_predictions))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print("Calculating AUC.")
    class_list = [i for i in range(num_classes)]
    labels_binarized = label_binarize(model_labels, classes=class_list)
    predictions_binarized = label_binarize(model_predictions, classes=class_list)

    print("Getting ROC curve. ")
    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(labels_binarized.ravel(), predictions_binarized.ravel())
    roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
    print(roc_auc['micro'])
    return float(roc_auc['micro'])


def get_aucs(models:list, dataset, num_workers:int=0):
    """Get AUCs for a list of models. 

    Args:
        models (list): _description_
        dataset (_type_): _description_
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    loader  = torch.utils.data.DataLoader(dataset,
                                            batch_size=256,
                                            shuffle=False,
                                            num_workers=num_workers)
    aucs = list()
    for index, model in enumerate(models):
        print(f"Running model... {index} ")
        auc = get_AUC(model=model, loader=loader)
        aucs.append(auc)
    
    return aucs

def metric_helper(models, metric_type:str, dataset=None, loader=None, num_workers:int=1):
    """Select which metric to use. 

    Args:
        models (_type_): Models. 
        metric_type (str): Metric Type. 
        dataset (_type_, optional): Which validation dataset to use. Defaults to None.
        loader (_type_, optional): Which dataloader to use. Defaults to None.
        num_workers (int, optional): Number of workers. Defaults to 1.

    Returns:
        list: List of validated metrics
    """
    if(metric_type == "AUC"):
        return get_aucs(models=models, dataset=dataset, num_workers=num_workers)
        #return get_AUC(model=model, dataset=dataset, loader=loader)