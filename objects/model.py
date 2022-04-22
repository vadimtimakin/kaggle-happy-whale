import math
import torch
import timm
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from convnext import *

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine   


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)    

    
class Backbone(nn.Module):

    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        # self.net = timm.create_model(name, pretrained=pretrained)
        self.net = convnext_large(pretrained=True, in_22k=True)

        # last_layer = list(self.net._modules)[-1]
        # try:
        #     self.out_features=getattr(self.net, last_layer).in_features
        # except AttributeError:
        #     self.out_features=getattr(self.net, last_layer)[1].in_features
        self.out_features = 1536

    def forward(self, x):
        x = self.net.forward_features(x)
        return x

    
class LesGoNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(LesGoNet, self).__init__()
        
        self.args = args
        self.backbone = Backbone(args.backbone, pretrained=pretrained)

        self.global_pool = GeM()

        self.embedding_size = args.embedding_size        
        
        self.neck = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
            nn.BatchNorm1d(self.embedding_size),
            torch.nn.PReLU()
        )
            
        self.head = ArcMarginProduct(self.embedding_size, args.n_classes)

    def forward(self, input_dict, get_embeddings=False):

        x = input_dict

        x = self.backbone(x)
        
        x = self.global_pool(x)
        
        x = x[:,:,0,0]

        x = self.neck(x)

        if get_embeddings:
            return x

        logits = self.head(x)

        return logits