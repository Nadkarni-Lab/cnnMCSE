"""Script to help with transfer learning. 
"""
import torch
import torch.nn as nn
from torchvision import models


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(pretrained=False, num_classes=1)
        base_model.load_state_dict(torch.load("/sc/arion/projects/EHR_ML/gulamf01/cnnMCSE/jobs/12_sota-update/12.1_sota-update/weights/RadImageNet-ResNet50_notop_torch.pth"))
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])
        for p in self.backbone.parameters():
            p.requires_grad = False
                        
    def forward(self, x):
        return self.backbone(x)

class TransferFeatures(nn.Module):
    def __init__(self, original_model):
        super(TransferFeatures, self).__init__()
        self.features = original_model.features
        print(self.features)
        for p in self.features.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        f = self.features(x)
        f = torch.flatten(f, start_dim=1)
        return f

def transfer_helper(transfer_base_model:str):
    """Method to generate pretrained model. 

    Args:
        transfer_base_model (str): String base models. 

    Returns:
        nn.Module: Model for a given zoo. 
    """
    if(transfer_base_model == "resnet18"):
        model = models.resnet18(pretrained=True)

    elif(transfer_base_model == "resnet50"):
        model = models.resnet50(pretrained=False)

    elif(transfer_base_model == "alexnet"):
        model = models.alexnet(pretrained=True)
        model = TransferFeatures(original_model=model)
        
    elif(transfer_base_model == "squeezenet"):    
        model = models.squeezenet1_0(pretrained=True)
    
    elif(transfer_base_model == "vgg16"):     
        model = models.vgg16(pretrained=True)

    elif(transfer_base_model == "densenet"):  
        model = models.densenet161(pretrained=True)

    elif(transfer_base_model == "inception_v3"): 
        model = models.inception_v3(pretrained=True)
    
    elif(transfer_base_model == "googlenet"): 
        model = models.googlenet(pretrained=True)
    
    elif(transfer_base_model == "shufflenet"):
        model = models.shufflenet_v2_x1_0(pretrained=True)

    elif(transfer_base_model == "mobilenet"):
        model = models.mobilenet_v3_large(pretrained=True)
    
    elif(transfer_base_model == "resnext50"):
        model = models.resnext50_32x4d(pretrained=True)
    
    elif(transfer_base_model == "wide_resnet50"):
        model = models.wide_resnet50_2(pretrained=True)
    
    elif(transfer_base_model == "mnasnet"):
        model = models.mnasnet1_0(pretrained=True)

    elif(transfer_base_model == "efficientnet_b7"):
        model = models.efficientnet_b7(pretrained=True)
    
    elif(transfer_base_model == "regnet_x_32gf"):
        model = models.regnet_x_32gf(pretrained=True)
    
    elif(transfer_base_model == "radimagenet"):
        model = Backbone()
       
    else:
        model = None
    
    return model