import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_l
from torchvision.models import mobilenet_v3_large
from torchvision.models import densenet161

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


import torch
from torchvision import datasets, models, transforms


class EFnet(nn.Module):
    def __init__(self,path=None):
        super(EFnet, self).__init__()
        self.backbone = efficientnet_v2_l(weights='IMAGENET1K_V1')
        
        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)

        self.layers = list(self.backbone.features.children())

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)

        return output
        

class MOnet(nn.Module):
    def __init__(self,path=None):
        super(MOnet, self).__init__()
        self.backbone = mobilenet_v3_large(weights='IMAGENET1K_V1')
        
        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
        
        self.layers = list(self.backbone.features.children())

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)

        return output



class DENSnet(nn.Module):
    def __init__(self,path=None):
        super(DENSnet, self).__init__()
        self.backbone = models.densenet201(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())
        self.layer_indices = [10]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                #print(output.shape)
                #outputs.append(output)
                #p += 1
                break

            layer_idx += 1

        return output

class INCEPTIONnet(nn.Module):
    def __init__(self,path=None):
        super(INCEPTIONnet, self).__init__()
        self.backbone = models.inception_v3(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())
        self.layer_indices = [8]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                break

            layer_idx += 1

        return output

class RESNET18net(nn.Module):
    def __init__(self,path=None):
        super(RESNET18net, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())
        self.layer_indices = [7]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                break

            layer_idx += 1

        return output

class RESNET18net(nn.Module):
    def __init__(self,path=None):
        super(RESNET18net, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())[:-1]
        self.layer_indices = [7]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                break

            layer_idx += 1

        return output


class RESNET50net(nn.Module):
    def __init__(self,path=None):
        super(RESNET50net, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())[:-1]
        self.layer_indices = [7]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                break

            layer_idx += 1

        return output

class RESNET101net(nn.Module):
    def __init__(self,path=None):
        super(RESNET101net, self).__init__()
        self.backbone = models.resnet101(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())[:-1]
        self.layer_indices = [7]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                break

            layer_idx += 1

        return output
        
class SHUFFLEnet(nn.Module):
    def __init__(self,path=None):
        super(SHUFFLEnet, self).__init__()
        self.backbone = models.shufflenet_v2_x2_0(weights='IMAGENET1K_V1')

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())[:-2]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)

        return output
        
class VITB16net(nn.Module):
    def __init__(self,path=None):
        super(VITB16net, self).__init__()
        self.backbone = models.vit_b_16(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.encoder.layers.children())[:-1] 
        self.layer_indices = [10]

    def forward(self, x):
        outputs = []
        output = x.squeeze(0)
        output = self.backbone.conv_proj(output)
        output = output.view(1, -1, 768)
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            
            layer_idx += 1
        
        return output.reshape(1,14,14,768).permute(0,3,1,2)

class VITL16net(nn.Module):
    def __init__(self,path=None):
        super(VITL16net, self).__init__()
        self.backbone = models.vit_l_16(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.encoder.layers.children())[:-1] 

    def forward(self, x):
        outputs = []
        output = x.squeeze(0)
        output = self.backbone.conv_proj(output)
        output = output.view(1, -1, 1024)
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            
            layer_idx += 1

        return output.reshape(1,14,14,1024).permute(0,3,1,2)

class SWINTnet(nn.Module):
    def __init__(self,path=None):
        super(SWINTnet, self).__init__()
        self.backbone = models.swin_v2_t(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output.permute(0,3,1,2)
        

class SWINBnet(nn.Module):
    def __init__(self,path=None):
        super(SWINBnet, self).__init__()
        self.backbone = models.swin_v2_b(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output.permute(0,3,1,2)

class SWINSnet(nn.Module):
    def __init__(self,path=None):
        super(SWINSnet, self).__init__()
        self.backbone = models.swin_v2_s(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output.permute(0,3,1,2)

class squeeznet(nn.Module):
    def __init__(self,path=None):
        super(squeeznet, self).__init__()
        self.backbone = models.squeezenet1_0(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output

class vgg19net(nn.Module):
    def __init__(self,path=None):
        super(vgg19net, self).__init__()
        self.backbone = models.vgg19_bn(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())[:-8]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output

class vgg16net(nn.Module):
    def __init__(self,path=None):
        super(vgg16net, self).__init__()
        self.backbone = models.vgg16_bn(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())[:-8]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output 

class vgg11net(nn.Module):
    def __init__(self,path=None):
        super(vgg11net, self).__init__()
        self.backbone = models.vgg11_bn(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.features.children())[:-6]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output 
    
    
class googlenet(nn.Module):
    def __init__(self,path=None):
        super(googlenet, self).__init__()
        self.backbone = models.googlenet(weights='IMAGENET1K_V1')   

        if path!=None:
            state_dict = torch.load(path)
                        
            # Get the current model's state dictionary
            model_state_dict = self.backbone.state_dict()
            
            for name, param in state_dict.items():
                
                if model_state_dict[name].shape == param.shape:
                    model_state_dict[name] = param
        
            self.backbone.load_state_dict(model_state_dict, strict=False)
                    
        self.layers = list(self.backbone.children())[:-11]

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            layer_idx += 1

        return output     
        
class EFnetUP(nn.Module):
    def __init__(self):
        super(EFnetUP, self).__init__()
        self.backbone = efficientnet_v2_m(weights='IMAGENET1K_V1')
        self.layers = list(self.backbone.features.children())
        self.layer_indices = [7]
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.backconv00 = nn.Conv2d(512, 1024, kernel_size=3, padding="same")
        self.backconv01 = nn.Conv2d(1024, 2048, kernel_size=3, padding="same")

        self.backconv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding="same")
        self.backconv2 = nn.Conv2d(1024, 512, kernel_size=3, padding="same")
        self.backconv3 = nn.Conv2d(512, 256, kernel_size=3, padding="same")

    def forward(self, x):
        outputs = []
        output = x
        layer_idx = 0
        p = 0

        for module in self.layers:
            output = module(output)
            if layer_idx in self.layer_indices:
                #print(output.shape)
                #outputs.append(output)
                break

            layer_idx += 1


        output = self.backconv00(output)
        output = self.backconv01(output)
        outputs.append(output)

        output = self.backconv1(output)
        output = self.upsample1(output)
        outputs.append(output)

        output = self.backconv2(output)
        output = self.upsample2(output)
        outputs.append(output)

        output = self.backconv3(output)
        output = self.upsample3(output)
        outputs.append(output)

        return outputs[::-1]
