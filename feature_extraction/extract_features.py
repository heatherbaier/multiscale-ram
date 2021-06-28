from utils import AverageMeter, load_inputs, get_png_names, get_ys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, utils, models
from model import RecurrentAttention
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import get_config
from trainer import Trainer
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision
import shutil
import pickle
import torch
import utils
import json
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


m = open("../../pooling/data/migration_data.json",)
mig_data = json.load(m)
m.close()
mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
mig_data.columns = ['muni_id', 'num_migrants']
q = 2
mig_data['class'] = pd.qcut(mig_data['num_migrants'], q = q, labels = [i for i in range(q)])
mig_data


image_names = get_png_names("../../attn/data/MEX/")

y_class, y_mig = get_ys(image_names, mig_data)


import matplotlib.pyplot as plt
import torchvision


import random

train_num = int(25 * .70)

train_indices = random.sample(range(0, 25), train_num)
val_indices = [i for i in range(0, 25) if i not in train_indices]


import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class miniConv(torch.nn.Module):
    
    def __init__(self, resnet):
        super().__init__()
        
        self.conv1_miniConv = resnet.conv1
        self.bn1_miniConv = resnet.bn1
        self.relu_miniConv = resnet.relu
        self.maxpool_miniConv = resnet.maxpool
        self.layer1_miniConv = resnet.layer1
        self.layer2_miniConv = resnet.layer2
        self.layer3_miniConv = resnet.layer3
        self.layer4_miniConv = resnet.layer4
        self.adp_pool_miniConv = torch.nn.AdaptiveAvgPool3d((512, 1, 1))
        
        D_in = 1 * 512 * 1 * 1
        h_g = 512
        self.fc1 = torch.nn.Linear(D_in, h_g)#.to('cuda:1')
        
    def forward(self, phi):
        
        phi = self.conv1_miniConv(phi)
        phi = self.bn1_miniConv(phi)
        phi = self.relu_miniConv(phi)
        phi = self.maxpool_miniConv(phi)
        phi = self.layer1_miniConv(phi)
        phi = self.layer2_miniConv(phi)
        phi = self.layer3_miniConv(phi)
        phi = self.layer4_miniConv(phi)
        phi = self.adp_pool_miniConv(phi)
        phi = phi.flatten(start_dim = 1)  
        phi = self.fc1(phi)
        return phi
    
    
resnet = torchvision.models.resnet18()
miniConv_model = miniConv(resnet = resnet)

# params = []
# for n,p in miniConv_model.named_parameters():
#     print(n)
#     params.append(n)
    
    
checkpoint = torch.load("./ckpt/ram_4_50x50_0.75_model_best.pth.tar")
checkpoint = checkpoint["model_state"]

matching_keys = [p for p in list(checkpoint.keys()) if "_miniConv" in p]
matching_keys2 = [p for p in list(checkpoint.keys()) if p == 'sensor.fc1.weight' or p == 'sensor.fc1.bias']
[matching_keys2.append(k) for k in matching_keys]

weights = {}
for mk in matching_keys2:
    miniConv_model_key = mk.replace("sensor.", "")
    weights[miniConv_model_key] = checkpoint[mk]
    
miniConv_model.load_state_dict(weights)


g = open("./locations_extract.json")
glimpses = json.load(g)
g.close()



features_dict = {}

for im_name, patches in glimpses.items():
    
    try:
        
        muni_id = im_name.split("/")[5]

        print(muni_id)

        image = load_inputs(im_name)

        cur_features_dict = {}

        for glimpseID, coords in patches.items():

            from_x, to_x, from_y, to_y = int(coords["from_x"]), int(coords["to_x"]), int(coords["from_y"]), int(coords["to_y"])

            cur_patch = image[:, :, from_x:to_x, from_y:to_y]

    #         plt.imshow(cur_patch[0].permute(1,2,0))
    #         plt.savefig(f"{muni_id}_{glimpseID}.png")

            features = miniConv_model(cur_patch)[0].detach().numpy()
            features = [str(f) for f in features]

            cur_features_dict[glimpseID] = features
            
    except:
        
        print("Skipping - bad to the bone.")

    features_dict[im_name] = cur_features_dict
            

with open("features_extract.json", "w") as outfile: 
    json.dump(features_dict, outfile)