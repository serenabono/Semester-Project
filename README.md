# Semester-Project

Image translation has been very successful in generating images with various domain shifts using
the popular CycleGAN framework. Few image translation models have also been able to generate
different deformations or properties of objects. In this semester project, we are interested in exploring
image understanding through image translation models. In particular we want to understand the pose
of the camera in order to automatically detect key image features of interest. We use the framework in
the context of Robocup football matches for understanding the football field. RoboCup is a scientific
initiative with the goal to advance the state of the art of intelligent robots able to play soccer,
originally aimed to reach human soccer World Cup champions level by 2050. The work presented
below suggests two possible unsupervised strategies for image translation from real to synthetic:
one based on GANs and the other based on pose regression and homography warping. The GAN
framework shows promising qualitative results where real data are converted to synthetic-like images.
The images can be easily thresholded to compute the regions of interest. We also show results of
direct pose regression on the translated images.

## Repo Contents
We propose two different methods of understanding poses and thus the scene through image translation:
1. Cycle-GAN + Posenet: The Cycle-GAN takes as inputs real images and synthetic images and
aims to translate the real images to the synthetic domain, in order to create a set of translated
images as similar as possible to the synthetic images. The synthetic images and their pose are fed
as training inputs to the Posenet. After that, the Posenet attempts to guess the position of the
translated images.
2. Posenet + Unsupervised Learning: The synthetic images and their pose are fed to the Posenet
as training inputs. After few epochs the Posenet attempts to to guess the position of the real images.
For each pose a synthetic image is generated and a loss is computed between the real and the newly
generated synthetic image.

## Cycle-GAN + Posenet
The code of the Cycle-GAN can be found after the homonym folder, while the Posnet architecture is found under the visloc-apr folder.

The Cycle-GAN was already implemented on GitHub. In order to start experimenting with the Cycle-
GAN, I had to generate synthetic images of the soccer field. To do so, it was enough to write a code
snippet (under the path /generate synthetic images/TrainBmain no robots.py)
that would generate homograpies of the top viewpoint of a soccer field. The task was eased by the fact
that soccer fields can be accurately defined by a 2D image and then projected using an image warp.
The missing details (i.e. the goal post depth/height) were not too relevant. The position of the camera,
irrelevant for this specific task, will be very important in the following, therefore it is saved in the
form of (d(x); d(y); d(z); q1; q2; q3; q4). The first three values represent the displacement in the respective
directions, while the last four are a particular measure of orientation named quaternions: 4-D values
easily mappable to legitimate rotations by normalization to unit length.
Training a Cycle-GAN is always a difficult task: tuning the hyperparameters such as images in-
put/output size and the learning rate are crucial to a correct implementation.

PosNet, the network to regress pose from images, was already implemented on GitHub. The architec-
ture was composed of an underlying core network, originally GoogLeNet, and a final fully connected layer
regressing the 6-dof pose. I changed the base network to Resnet34, already implemented on GitHub
(at path "/visloc-apr"). As explained in the Posenet paper, in order to regress pose, the convnet was trained on Euclidean loss using stochastic gradient descent

##  Posenet + Unsupervised Learning
For the Unsupervised module the Posenet is slightly modified, the code can be found under the new architecture folder.
Looking back at the performance of the previous method, it was clear that my implementation of Posenet had an issue:
the network was trained on synthetic images, but tested on Cycle-GAN translated images.
Furthermore the synthetic images provided to the network were generated with similar focal length and
similar displacement along y (height), but the dataset of real images contained a lot of more variability.
Most of the images could not be correctly processed because there was no correspondence to their real
pose in the network training data. To wave this issue, I introduced a self-supervised module. (GitHub
repository under the path /new-architecture/ )

The loss function was a very delicate matter. As matter of facts the images to be compared did come
from two very different domains. The main objective was to take into consideration only the features
truly important for the localization, otherwise the variability of the real images (presence or absence of
ball/robots/public, lighting etc ...) would completely overtake the loss function. The images had to be
preprocessed using difierentiable tools such as filters. Few of the different combinations were:
1. MSE Loss
2. L1 Loss
3. L1 Loss + 2D Gaussian Blur + Sobel Filter: The gaussian blur smoothed out the real images
and made them more uniform, while the Sobel filter underlined the contour of the figures and
therefore the lines of the field.

## Conclusions

The self-supervised module was successful in predicting from real images regardless of their intrinsic
variability. The results show how the focal length and the height varies within the predictions. The
model is still not perfect, lots of images are off, probably because the loss functions found are still not
the best possible choice in detecting the main features through domains. If I were to keep working on
this project I would probably spend some time in searching more effective loss functions and in testing
their accuracy both on the real images and Cycle-GAN translated images.

## Results can be seen the the paper pdf


## Citations

> -- <cite>[cycleGAN-PyTorch][2]</cite>
https://github.com/arnab39/cycleGAN-PyTorch

> -- <cite>[visloc-apr][3]</cite>
https://github.com/GrumpyZhou/visloc-apr
