from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value
from model import RecurrentAttention
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import AverageMeter
from config import get_config
from utils import plot_images
from trainer import Trainer
from tqdm import tqdm
import pandas as pd
import torchvision
import numpy as np
import random
import pickle
import shutil
import torch
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from utils import *


q = 4
with open("../../pooling/data/migration_data.json") as m:
    mig_data = json.load(m)
m.close()
mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
mig_data.columns = ['muni_id', 'num_migrants']
mig_data['class'] = pd.qcut(mig_data['num_migrants'], q = q, labels = [i for i in range(q)])
mig_data.head()


image_names = get_png_names("../../attn/data/MEX/")
y_class, y_mig = get_ys(image_names, mig_data)


with open("../val_images.txt") as ims:
    val_names = ims.read().splitlines()
    
train_names = [i for i in image_names if i not in val_names]

val_indices = [image_names.index(i) for i in val_names]
train_indices = [image_names.index(i) for i in train_names]
train_num = int(len(train_indices))


# train_num = int(25 * .70)
# train_indices = random.sample(range(0, 25), train_num)
# val_indices = [i for i in range(0, 25) if i not in train_indices]


# train_num = int(len(image_names) * .70)
# train_indices = random.sample(range(0, len(image_names)), train_num)
# val_indices = [i for i in range(0, len(image_names)) if i not in train_indices]


# validation_images = [image_names[i] for i in val_indices]
# with open('val_images.txt', 'w') as f:
#     for item in validation_images:
#         f.write("%s\n" % item)


batch_size = 1
train = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y_class[i], y_mig[i]) for i in train_indices]
val = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y_class[i], y_mig[i]) for i in val_indices]
train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)


print("Num training: ", len(train_dl))
print("Num validation: ", len(val_dl))


config, unparsed = get_config()
trainer = Trainer(config, (train_dl, val_dl))
trainer.train()

