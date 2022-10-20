import os
import numpy as np
import torch
import torch.nn as nn
import model.resnet.resnet as resnet
from model.PCL.resnet_pcl import resnet18

def load_CRL_encoder(obs_space):
    resnet_baseplanes = 64
    backbone = "resnet50"
    hidden_size = 512

    visual_resnet = resnet.ResNetEncoder(
        obs_space,
        baseplanes=resnet_baseplanes,
        ngroups=resnet_baseplanes // 2,
        make_backbone=getattr(resnet, backbone),
        normalize_visual_inputs=False,
        spatial_shape=[256,256]
    )

    visual_encoder = nn.Sequential(
        visual_resnet,
        nn.Flatten(),
        nn.Linear(
            np.prod(visual_resnet.output_shape), hidden_size
        ),
        nn.ReLU(True),
    )
    #input(visual_resnet.state_dict().keys())
    ckpt_pth = os.path.join('model/CRL', 'CRL_pretrained_encoder_mp3d.pth')
    state_dict = torch.load(ckpt_pth, map_location='cpu')['state_dict']

    # from copy import deepcopy
    # new_state_dict = deepcopy(state_dict)
    # for k,v in state_dict.items():
    #     v = new_state_dict.pop(k)
    #     if "model_encoder.backbone." in k:
    #         split_idx = k.find("model_encoder.backbone.") + len("model_encoder.backbone.")
    #         new_state_dict[k[split_idx:]] = v 

    # torch.save({'state_dict':new_state_dict}, '/home/hongxin_li/hongxin_li@172.18.33.10/Github/EmbodiedUniT/model/CRL/CRL_pretrained_encoder_mp3d.pth')

    visual_encoder[0].backbone.load_state_dict(state_dict) # only load the ResNet50 backbone
    
    print("\n\033[0;33;40m[encoder_loaders] CRL pretrained model loaded\033[0m\n")

    return visual_encoder

def load_PCL_encoder(feature_dim):
    visual_encoder = resnet18(num_classes=feature_dim)
    dim_mlp = visual_encoder.fc.weight.shape[1]
    visual_encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), visual_encoder.fc)
    ckpt_pth = os.path.join('model/PCL', 'PCL_encoder.pth')
    ckpt = torch.load(ckpt_pth, map_location='cpu')
    visual_encoder.load_state_dict(ckpt)

    print("\n\033[0;33;40m[encoder_loaders] PCL pretrained model loaded\033[0m\n")
    return visual_encoder