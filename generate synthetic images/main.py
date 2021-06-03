import h5py
import io
from PIL import Image
import numpy as np
import imageio


#'013-RoboCup-2019-Outdoor-Field-2/SydneyOutdoor_rUNSWift_1stHalf/1-Sarah/'

data = []  # list all images files full path 'group/subgroup/b.png' for e.g. ./A/a/b.png. These are basically keys to access our image data.

group = [] # list all groups and subgroups in hdf5 file

def func(name, obj):     # function to recursively store all the keys
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

save_path="fieldboundary.hdf5"
hf = h5py.File(save_path, 'r')
hf.visititems(func)  # this is the operation we are talking about.

# Now lets read the image files in their proper format to use it for our training.
final=[]
#for d in data:
#    if("RoboCup" in d and "images" in d):
#        final.append(d)

i=0
for j in data:
    dset = hf[j]
    datas=np.asarray(dset.value)
    for k in range(len(datas)):
        try:
            img = Image.open(io.BytesIO(datas[k]))
            img.save(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/realdata/{i}.jpg", "JPEG", quality=80, optimize=True, progressive=True)
            i=i+1
        except:
            print(f"image {i}, file {j} not printed")

hf.close()

