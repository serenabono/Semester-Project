import numpy as np
import cv2 as cv2
from skimage import transform
from skimage.io import imread, imshow
from scipy.spatial.transform import Rotation as R
from PIL import Image

def runSimulate(K,dx,dy,dz,q1,q2,q3,q4):
    T=np.asarray([dx,dy,dz])
    R_f = R.from_quat([[q1,q2,q3,q4]])
    R_f=R_f.as_matrix()
    H=np.column_stack((R_f[0,:,0:2],T*10))
    H=np.matmul(K,H)
    return H

def simImage(img,H):
    tform = transform.ProjectiveTransform(H)
    img2 = transform.warp(img, tform.inverse, mode="constant")

    return img2

def main():
    faketxt = open('predicted.txt', 'r')

    for i, fake in enumerate(faketxt.readlines()):
        fake_split=fake.split(" ")
        img = imread("/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/robocup_thicker.jpeg")
        row, col, ch = img.shape
        zoom = 1300
        K = np.asarray([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]], dtype=np.float64)
        H_fake = runSimulate(K,float(fake_split[0]),float(fake_split[1]),float(fake_split[2]),float(fake_split[3]),float(fake_split[4]),float(fake_split[5]),float(fake_split[6]))
        img_out_real=Image.open(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/only-converted/{i}.png")
        img_out_fake = simImage(img, H_fake)
        img_out = np.concatenate((img_out_real.resize((img_out_fake.shape[1],img_out_fake.shape[0]), Image.ANTIALIAS), img_out_fake*255), axis=1)
        cv2.imwrite(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/reconstructed-real/{i}.png", img_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

if __name__=="__main__":
    main()
