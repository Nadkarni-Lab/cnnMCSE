"""Script to help with transfer learning. 
"""
from torchvision import models

def transfer_helper(transfer_base_model:str):
    if(transfer_base_model == "resnet18"):
        model = models.resnet18(pretrained=True)

    elif(transfer_base_model == "alexnet"):
        model = models.alexnet(pretrained=True)

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
    
    else:
        model = None
    
    return model