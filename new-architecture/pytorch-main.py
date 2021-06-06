import random
import cv2 as cv2
import numpy as np
import torch
import torchgeometry as tgm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

def simImage(img, H):
    homography = H.view(1, 3, 3)
    img=tgm.warp_perspective(img, homography, dsize=(img.shape[2], img.shape[3]))
    return img


def runSimulate(K, dx, thetax, dy, thetay, dz, thetaz):
    T = torch.tensor([dx, dy, dz],requires_grad=True)
    Rx = torch.tensor([[1, 0, 0], [0, np.cos(thetax), -np.sin(thetax)], [0, np.sin(thetax), np.cos(thetax)]],requires_grad=True)
    Ry = torch.tensor([[np.cos(thetay), 0, np.sin(thetay)], [0, 1, 0], [-np.sin(thetay), 0, np.cos(thetay)]],requires_grad=True)
    Rz = torch.tensor([[np.cos(thetaz), -np.sin(thetaz), 0], [np.sin(thetaz), np.cos(thetaz), 0], [0, 0, 1]],requires_grad=True)
    R_f = torch.matmul(torch.matmul(Rx, Ry), Rz).type(torch.FloatTensor)
    R_ff = torch.transpose(torch.cat((torch.transpose(R_f,1,0), torch.unsqueeze(torch.tensor([0,0,0]).type(torch.FloatTensor),dim=0)),dim=0),1,0)
    quaternion = tgm.rotation_matrix_to_quaternion(torch.unsqueeze(R_ff,dim=0))
    label = torch.cat((torch.unsqueeze(torch.transpose(T,-1,0),dim=0) / 10, quaternion),dim=1)
    H = torch.cat((R_ff[:, 0:2], torch.unsqueeze(T,dim=1)),dim=1).type(torch.FloatTensor)
    H = torch.matmul(K, H)
    H=H/H[2,2]
    return H, label


def main():
    def show(img):
        npimg = img.detach().numpy().reshape([3,842,1194])
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
        plt.show()

    frontal = True
    labeltxt = open('generated-images-2.txt', 'w')
    im_path = '/itet-stor/sebono/net_scratch/visloc-apr-new-architecture/robocup_thicker.jpeg'
    image = Image.open(im_path)
    image = image.convert('RGB')

    im_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = im_transform(image)
    # Add batch dimension


    img = image.unsqueeze(dim=0)
    if(frontal==True):
        folder="/itet-stor/sebono/net_scratch/visloc-apr-new-architecture/generated-images"
        x=-np.pi/2-np.pi/2-np.pi/6
        _,ch,row, col = img.shape
        zoom = 1900
        rangex=np.linspace(100,-row/1.2,24)
        rangey = np.linspace(200, -col/1.2, 24)
        rangea=np.asarray(np.linspace(0, np.pi/2.5, 20))
    else:
        folder="/itet-stor/sebono/net_scratch/visloc-apr-new-architecture/generated-images"
        x=-np.pi/2-np.pi/6
        _,ch,row, col = img.shape
        zoom=1900
        rangex=np.linspace(100,-1400,24)
        rangey = np.linspace(100, -700, 24)
        rangea=np.asarray(np.linspace(0, np.pi, 20))

    i = 1464
    random.seed(43)
    for el1 in rangex:
        for el2 in rangey:
            for el in rangea:
                K = torch.tensor([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]]).type(torch.FloatTensor)
                H, label = runSimulate(K, el2, -np.pi / 2, 100 - np.pi / 8, 0, el1, x+el)
                img_out = simImage(img, H)
                thresh = int(img.shape[0] / 2)
                # remove reflection
                img_out[:thresh, :, :] = torch.zeros(img[:thresh, :, :].shape)
                a = torch.sum(img_out.reshape([img.shape[1] * img_out.shape[2], img_out.shape[3]]), axis=1) == 0
                tot_black = torch.sum(a)

                # filter out images with more the 70% of black
                if (tot_black > 0.7 * img_out.shape[1] * img_out.shape[2]):
                    continue

                df = pd.DataFrame(label.detach().numpy()).transpose()
                img_out[:thresh, :, :] = torch.zeros(img_out[:thresh, :, :].shape)
                cv2.imwrite(folder + f"/{i}.png", np.transpose(img_out.detach().numpy().reshape([3,842,1194]) * 255,(1, 2, 0)), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                labeltxt.write(f"generated-images/{i}.png  "+df.to_string(header=False, index=False)+"\n")
                i += 1

    return


if __name__ == "__main__":
    main()
