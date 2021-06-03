import PIL.Image
import os


folder="/itet-stor/sebono/net_scratch/visloc-apr/data/Robocup-n/seq2"

def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        rgba_image = PIL.Image.open(os.path.join(folder,filename))
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save(os.path.join(folder,filename), 'PNG')

load_images_from_folder(folder)