import h5py
import io
from PIL import Image
import numpy as np
import imageio
import numpy as np

save_path="fieldboundary.hdf5"
hf = h5py.File(save_path, 'r')
dset = hf.get("101-UERoboCup-Synthetic/images")
datas=np.asarray(dset.value)
i=0
for k in range(len(datas)):
    try:
        img = Image.open(io.BytesIO(datas[k]))
        img.save(f"/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/syntheticdata/{i}.jpg", "JPEG", quality=80, optimize=True, progressive=True)
        i=i+1
    except:
        print(f"image {i}, file  not printed")