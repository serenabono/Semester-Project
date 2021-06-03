import random, os
"""
#REAL IMAGES
path = "./realdata/"
#create Train Dataset
for i in range(0,2500):
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    os.rename(path+random_filename, "./trainA/"+random_filename)
print("finished with TrainA")
#create Test Dataset
for i in range(150):
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    os.rename(path+random_filename, "./testA/"+random_filename)
print("finished with TestA")
"""
label_prev1 = open('labels_seq1.txt', 'r')
label_prev2 = open('labels_seq2.txt', 'r')
#SYNTHETIC IMAGES
path = "/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/synthetic_images_thick/"
#create Train Dataset
labeltxt = open('dataset_train.txt', 'w')
lines1=label_prev1.readlines()
lines2=label_prev2.readlines()

for i in range(0,3000):
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    name=int(random_filename.split(".")[0])
    if(name> 2303):
        name-=2303
        labeltxt.write(lines1[name])
    else:
        labeltxt.write(lines2[name])
    #os.rename(path+random_filename, "/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/trainB-wrobot/"+random_filename)

print("finished with TrainB")
labeltxt.close()

labeltxt = open('dataset_test.txt', 'w')
#create Test Dataset
for i in range(0,300):
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    name=int(random_filename.split(".")[0])
    if(name> 2303):
        name-=2303
        labeltxt.write(lines1[name])
    else:
        labeltxt.write(lines2[name])

    #os.rename(path+random_filename, "/itet-stor/sebono/net_scratch/datasets/fieldboundary/images/testB-wrobot/"+random_filename)

print("finished with TestB")
labeltxt.close()

label_prev1.close()
label_prev2.close()