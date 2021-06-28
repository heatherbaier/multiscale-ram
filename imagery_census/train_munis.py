from utils import AverageMeter, load_inputs, get_png_names, get_ys, get_census
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, utils, models
from model import RecurrentAttention
import torch.nn.functional as F
from config import get_config
from trainer import Trainer
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision
import random
import shutil
import pickle
import torch
import utils
import json
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


m = open("../../pooling/data/migration_data.json",)
mig_data = json.load(m)
m.close()

c = open("../../archive/CAOE/data/census_data.json",)
census = json.load(c)
c.close()  


mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
mig_data.columns = ['muni_id', 'num_migrants']
q = 4
mig_data['class'] = pd.qcut(mig_data['num_migrants'], q = q, labels = [i for i in range(q)])
mig_data


image_names = get_png_names("../../attn/data/MEX/")
y_class, y_mig = get_ys(image_names, mig_data)
census_data = get_census(image_names, census)


train_num = int(len(image_names) * .70)
train_indices = random.sample(range(0, len(image_names)), train_num)
val_indices = [i for i in range(0, len(image_names)) if i not in train_indices]


# train_num = int(25 * .70)
# train_indices = random.sample(range(0, 25), train_num)
# val_indices = [i for i in range(0, 25) if i not in train_indices]


batch_size = 1
train = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y_class[i], y_mig[i], torch.tensor(census_data[i])) for i in train_indices]
val = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y_class[i], y_mig[i], torch.tensor(census_data[i])) for i in val_indices]
train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)


config, unparsed = get_config()
trainer = Trainer(config, (train_dl, val_dl))
trainer.train()