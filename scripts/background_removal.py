import sys
sys.path.append('/home/toefl/K/dolphin/U-2-Net')

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from model import U2NET 
from model import U2NETP 
from tqdm import tqdm

from IPython.display import display
from PIL import Image as Img

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

train_df = pd.read_csv("/home/toefl/K/dolphin/datasets/happy-whale-and-dolphin/train.csv")

input_path = "/home/toefl/K/dolphin/datasets/backfins/train_images"
new_path = '/home/toefl/K/dolphin/datasets/masks/train'

img_to_draw = train_df.image.values
print(len(img_to_draw))

THRESHOLD = 0.3

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def pred_unet(model, imgs):
    
    test_salobj_dataset = SalObjDataset(img_name_list = imgs, lbl_name_list = [], transform = transforms.Compose([RescaleT(320),ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers = 1)
    
    for i_test, data_test in enumerate(test_salobj_dataloader):
        
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        predict = d5[:,0,:,:]
        predict = normPRED(predict)
        
        del d1, d2, d3, d4, d5, d6, d7

        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()

        # Masked image - using threshold you can soften/sharpen mask boundaries
        predict_np[predict_np > THRESHOLD] = 1
        predict_np[predict_np <= THRESHOLD] = 0
        mask = Img.fromarray(predict_np*255).convert('RGB')
        image = Img.open(imgs[0])
        imask = mask.resize((image.width, image.height), resample=Img.BILINEAR)
        back = Img.new("RGB", (image.width, image.height), (255, 255, 255))
        mask = imask.convert('L')
        im_out = Img.composite(image, back, mask)
        
        # Sailient mask 
        salient_mask = np.array(image)
        mask_layer = np.array(imask)        
        mask_layer[mask_layer == 255] = 50 # offest on RED channel
        salient_mask[:,:,0] += mask_layer[:,:, 0]
        salient_mask = np.clip(salient_mask, 0, 255) 
    
    return np.array(im_out), np.array(image), np.array(salient_mask), np.array(mask)

UNET2_SMALL = False

if UNET2_SMALL:
    model_dir = "./U-2-Net/u2netp.pth"  # Faster ... a lot (!) but less accurate
    net = U2NETP(3,1) 
else:
    model_dir = "/home/toefl/K/dolphin/U-2-Net/u2net.pth"
    net = U2NET(3,1) 


if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_dir))
    net.cuda()
else:        
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

net.eval()

for idx, img in enumerate(tqdm(img_to_draw)):
    _, _, _, mask = pred_unet(net, [os.path.join(input_path, img_to_draw[idx])]) 
    mask = mask.astype(float)
    cv2.imwrite(os.path.join(new_path, img_to_draw[idx]), mask)
    