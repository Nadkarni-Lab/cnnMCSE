"""Metrics.py contains all the metrics for machine learning validation. 
"""
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize

def get_AUC(model, dataset, num_workers:int=2, num_classes:int=10):
    """Get Area-Under-Curve Metric on a test dataset. 

    Args:
        model (_type_): Model to validate. 
        dataset (_type_): Dataset to use. 

    Returns:
        float: AUC metric. 
    """

    # print('Testset:',len(testset))
    loader  = torch.utils.data.DataLoader(dataset,
                                          batch_size=len(dataset),
                                          shuffle=False,
                                          num_workers=num_workers)

    # current_model = FCN()

    # current_model = nn.DataParallel(current_model)
    # current_model.to(DEVICE)

    # current_model.load_state_dict(model_state)

    # current_model.eval()

    current_model = model
    current_model.eval()

    model_predictions = []
    model_labels      = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = torch.flatten(images, start_dim=1)

            # images, labels = images.to(DEVICE), labels.to(DEVICE)
            output = current_model(images)
            _, predicted = torch.max(output.data, 1)
        for prediction in predicted:
            model_predictions.append(prediction.tolist())
        for label in labels:
            model_labels.append(label.tolist())

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    class_list = [i for i in range(num_classes)]
    labels_binarized = label_binarize(model_labels, classes=class_list)
    predictions_binarized = label_binarize(model_predictions, classes=class_list)

    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(labels_binarized.ravel(), predictions_binarized.ravel())
    roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])
    return roc_auc['micro']


def metric_helper(model, dataset, metric_type:str):
    if(metric_type == "AUC"):
        return get_AUC(model, dataset)