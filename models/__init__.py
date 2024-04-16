from .ResNetA import ResNetA
from .plainNet import plainNet
from .sub_CNN import sub_CNN

def build_model(model_name,config,groups):
    if model_name =='resnet':
        model = ResNetA(config)
    elif model_name == 'plain_net': 
        model = plainNet(config)
    elif model_name == 'sub_CNN':
        model = sub_CNN(config,groups);
    
    return model
    