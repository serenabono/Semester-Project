import random
import cv2 as cv2
import numpy as np
import torch
import torchgeometry as tgm
from skimage.io import imread
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def simImage(img, H):
    homography = H.view(1, 3, 3)
    img=tgm.warp_perspective(img, homography, dsize=(img.shape[2], img.shape[3]))
    return img


def runSimulate(K,fake):
    T = fake[0:3]
    angle_axis = tgm.quaternion_to_angle_axis(fake[3:])
    R_f=tgm.angle_axis_to_rotation_matrix(torch.unsqueeze(angle_axis,dim=0))
    H = torch.cat((R_f[0,0:3, 0:2], torch.unsqueeze(T,dim=1)),dim=1).type(torch.Tensor)
    H = torch.matmul(K, H)
    H=H/H[2,2]
    return H


def evaluate_images(xyz,wpqr):
    coordinates=torch.cat((xyz,wpqr), dim=1)
    images=torch.zeros([len(coordinates),3,224,224])
    im_path = '/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/robocup_thicker.jpeg'
    image = Image.open(im_path)
    image = image.convert('RGB')

    im_transform = transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(1,1), ratio=(0.75, 1.3333333333333333), interpolation=1),
        transforms.ToTensor()
    ])

    img = torch.unsqueeze(im_transform(image),dim=0)
    _,ch,row, col = img.shape

    def show(img):
        npimg = img.detach().numpy().reshape([3,224,224])
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        plt.show()

    for i, fake in enumerate(coordinates):
        zoom = 1600
        K = torch.tensor([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]],requires_grad=True).type(torch.Tensor)
        H = runSimulate(K,fake)
        image_out=simImage(img,H)
        #noise=torch.repeat_interleave(torch.repeat_interleave(torch.linspace(0,1,steps=image_out.shape[3]),3),image_out.shape[2]).reshape(image_out.shape[1],image_out.shape[2],image_out.shape[3])
        #image_out=image_out*torch.unsqueeze(noise,dim=0)
        #show(image_out)
        images[i,:,:,:]=image_out[0]
    return images
