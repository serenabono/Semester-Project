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

    #img=np.rot90(img)
    #generate 3 random robots
    def gen_robots(nrobots=3,robot_dima=int(img.shape[0]/50),robot_dimb=int(img.shape[0]/50)):
        robot_coord= np.zeros([nrobots,3])

        for i in range(nrobots):
            robot_coord[i,0]=random.randint(0+int(robot_dimb/2), img.shape[0]-int(robot_dimb/2))
            robot_coord[i,1]=random.randint(0+int(robot_dimb/2), img.shape[1]-int(robot_dimb/2))
            robot_coord[i,2]=int(robot_dimb/2)
        return robot_coord

    if(frontal==True):
        #frontal: remember to put image frontal
        row, col, ch = img.shape
        zoom = 1600
        rangex=np.linspace(100,-row/1.2,24)
        rangey = np.linspace(200, -col/1.2, 24)
        rangea=np.asarray(np.linspace(0, np.pi/2.5, 20))
        rangez=np.asarray(np.linspace(0, np.pi/4, 5))
    else:
        #lateral: remember to put image lateral
        row,col,ch=img.shape
        zoom=1600
        rangex=np.linspace(100,-1400,24)
        rangey = np.linspace(100, -700, 24)
        rangea=np.asarray(np.linspace(np.pi/3, np.pi, 20))

    #-np.pi/2
    img_robot = cv2.imread('/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/robot-yellow.png', flags=cv2.IMREAD_UNCHANGED)
    i=2304
    random.seed(43)
    for el1 in rangex:
        for el2 in rangey:
            for el in rangea:
                nrobots = 3
                robot_dima = int(img.shape[1] / 10)
                robot_dimb = int(img.shape[1] / 10)
                coord=gen_robots(nrobots=nrobots,robot_dima=robot_dima,robot_dimb=robot_dimb)
                #dx,thetax,dy,thetay,dz,thetaz
                K = np.asarray([[zoom, 0, col / 2], [0, zoom, row / 2.5], [0, 0, 1]], dtype=np.float64)
                H,label = runSimulate(K, el2, -np.pi / 2, 100-np.pi/8, 0, el1, -np.pi/2-np.pi/2-np.pi/6+el)

                img_out = simImage(img, H)

                # img_out[] = [0.5, 0.5, 0.5]
                thresh = int(img.shape[0] / 2)
                # remove reflection
                img_out[:thresh,:,:]=np.zeros(img[:thresh,:,:].shape)
                a = np.sum(img_out.reshape([img.shape[0] * img_out.shape[1], img_out.shape[2]]), axis=1) == 0
                tot_black = np.sum(a)
                if (tot_black > 0.7 * img_out.shape[0] * img_out.shape[1]):
                    continue
                
                #remove comment if you want to generate also robots within the image
                """
                b_channel, g_channel, r_channel = cv2.split(img_out)
                alpha_channel = np.ones(b_channel.shape,
                                        dtype=b_channel.dtype)  # creating a dummy alpha channel image.
                img_out = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    
                for n in range(nrobots):
                    #[39,137,0]
                    fimg = img.copy()
                    dim_square=4
                    fimg[int(coord[n,0]),int(coord[n,1]),:]=np.ones([1,1,3])*255
                    fimg[int(coord[n,0])-dim_square:int(coord[n,0])+dim_square,int(coord[n,1])-dim_square:int(coord[n,1])+dim_square,:]=np.ones(img[int(coord[n,0])-dim_square:int(coord[n,0])+dim_square,int(coord[n,1])-dim_square:int(coord[n,1])+dim_square,:].shape)*255
                    fimg[int(coord[n,0]+coord[n,2])-dim_square:int(coord[n,0]+coord[n,2])+dim_square,int(coord[n,1])-dim_square:int(coord[n,1])+dim_square,:]=np.ones(fimg[int(coord[n,0]+coord[n,2])-dim_square:int(coord[n,0]+coord[n,2])+dim_square,int(coord[n,1])-dim_square:int(coord[n,1])+dim_square,:].shape)*255
                    K_rob = np.asarray([[zoom, 0, col / 2], [0, zoom, int(row / 2)*1.4], [0, 0, 1]], dtype=np.float64)
                    H_rob,_ = runSimulate(K_rob, el2, -np.pi / 2, 100, 0, el1, -np.pi/2-np.pi/6+el)
                    img_out0=simImage(fimg-img, H_rob)
                    img_out2=simImage(fimg-img, H)
                    img_out0 = np.nan_to_num(img_out0)
                    img_out2 = np.nan_to_num(img_out2)
    
                    if((len(np.argwhere(img_out0[:, :, 1] != 0))!=0)&(len(np.argwhere(img_out2[:, :, 1] != 0))!=0)):
                        arr_height=np.argwhere(img_out0[:, :, 1] != 0)
                        arr_width=np.argwhere(img_out2[:, :, 1] != 0)
                        max_x=max(arr_height[:,0])
                        max_y=max(arr_width[:,1])
                        min_x=min(arr_width[:,0])
                        min_y=min(arr_width[:,1])
                        try:
                            img_robotr = cv2.resize(img_robot, dsize=(int(max_y-min_y),int(max_x-min_x)), interpolation=cv2.INTER_CUBIC)
                            img_out[min_x:max_x,min_y:max_y,:][img_robotr[:,:,3]!=0]=img_robotr[img_robotr[:,:,3]!=0]/255
                        except:
                            pass
                """
                #comment for robots
                img_out[:thresh,:,:]=np.zeros(img_out[:thresh,:,:].shape)
                
                #remove comment for robots
                #img_out[:thresh,:,3]=np.ones(img_out[:thresh,:,3].shape)
                
                img_out=np.nan_to_num(img_out)
               
                #comment for robots
                img_out=img_out+np.repeat(np.repeat(np.linspace(1,0,img_out.shape[1]),3),img.shape[0]).reshape(img_out.shape)
                cv2.imwrite(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/seq1/{i}.png", img_out*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
                #remove comment for robots
                #img_out=img_out*255+np.repeat(np.repeat(np.linspace(1,0,img_out.shape[1]),4),img.shape[0]).reshape(img_out.shape)
                #cv2.imwrite(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/seq1/{i}.png", img_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                df = pd.DataFrame(label).transpose()

                if frontal==True:
                    labeltxt.write(f"seq1/{i}.png  "+df.to_string(header=False, index=False)+"\n")
                else:
                    labeltxt.write(f"seq2/{i}.png  "+df.to_string(header=False, index=False)+"\n")
                i +=1
                if i%100==0:
                    print(i)

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
