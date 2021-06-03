import numpy as np
import cv2 as cv2
from skimage import transform
from skimage.io import imread, imshow
import random
from scipy.spatial.transform import Rotation as R
import pandas as pd
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
def simImage(img,H):
    tform = transform.ProjectiveTransform(H)
    img2 = transform.warp(img, tform.inverse, mode="constant")
    return img2


def runSimulate(K,dx,thetax,dy,thetay,dz,thetaz):
    T=np.asarray([dx,dy,dz])
    Rx=[[1,0,0],[0,np.cos(thetax),-np.sin(thetax)],[0,np.sin(thetax),np.cos(thetax)]]
    Ry = [[np.cos(thetay),0,np.sin(thetay)],[0,1,0],[-np.sin(thetay),0,np.cos(thetay)]]
    Rz = [[np.cos(thetaz),-np.sin(thetaz),0],[np.sin(thetaz),np.cos(thetaz),0],[0,0,1]]
    R_f=np.matmul(np.matmul(Rx,Ry),Rz)
    quaternion = R.from_matrix(R_f).as_quat()
    label=np.concatenate((T/10,quaternion))
    H=np.column_stack((R_f[:,0:2],T))
    H=np.matmul(K,H)
    return H,label


def main():
    labeltxt = open('labels_seq1.txt', 'w')
    frontal = True
    img = imread("/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/robocup_thicker.jpeg")

    if(frontal==True):
        x=-np.pi/2-np.pi/2-np.pi/6+el
        row, col, ch = img.shape
        zoom = 1600
        rangex=np.linspace(100,-row/1.2,24)
        rangey = np.linspace(200, -col/1.2, 24)
        rangea=np.asarray(np.linspace(0, np.pi/2.5, 20))
        rangez=np.asarray(np.linspace(0, np.pi/4, 5))
    else:
        x=-np.pi/2-np.pi/6+el
        row,col,ch=img.shape
        zoom=1600
        rangex=np.linspace(100,-1400,24)
        rangey = np.linspace(100, -700, 24)
        rangea=np.asarray(np.linspace(np.pi/3, np.pi, 20))

    i=0
    random.seed(43)
    for el1 in rangex:
        for el2 in rangey:
            for el in rangea:

                K = np.asarray([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]], dtype=np.float64)
                H,label = runSimulate(K, el2, -np.pi / 2, 100-np.pi/8, 0, el1, x)

                img_out = simImage(img, H)

                thresh = int(img.shape[0] / 2)
                # remove reflection
                img_out[:thresh,:,:]=np.zeros(img[:thresh,:,:].shape)
                a = np.sum(img_out.reshape([img.shape[0] * img_out.shape[1], img_out.shape[2]]), axis=1) == 0
                tot_black = np.sum(a)
                #filter out images with more the 70% of black
                if (tot_black > 0.7 * img_out.shape[0] * img_out.shape[1]):
                    continue

                img_out[:thresh,:,:]=np.zeros(img_out[:thresh,:,:].shape)
                img_out=np.nan_to_num(img_out)
                img_out=img_out+np.repeat(np.repeat(np.linspace(1,0,img_out.shape[1]),3),img.shape[0]).reshape(img_out.shape)
                cv2.imwrite(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/seq1/{i}.png", img_out*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                df = pd.DataFrame(label).transpose()

                if frontal==True:
                    labeltxt.write(f"seq1/{i}.png  "+df.to_string(header=False, index=False)+"\n")
                else:
                    labeltxt.write(f"seq2/{i}.png  "+df.to_string(header=False, index=False)+"\n")
                i +=1

    labeltxt.close()
    return
#image gray
#size_x=img_out.shape[0]
#size_y=img_out.shape[1]
#img_out=img_out.reshape([size_x*size_y,img_out.shape[2]])
#img_out[a,:]=[0.5,0.5,0.5]
#img_out=img_out.reshape([size_x, size_y, 3])

if __name__=="__main__":
    main()
