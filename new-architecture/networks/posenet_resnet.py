import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base.basenet import BaseNet
from networks.base.resnet import resnet34
from networks.base.evaluate import evaluate_images
import kornia as K
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchgeometry.losses import ssim
from torchvision import transforms, datasets
from utils.common.setup import lprint
from .utils import cuda

class Regression(nn.Module):
    """Pose regression module.
    Args:
        regid: id to map the length of the last dimension of the input
               feature maps.
        with_embedding: if set True, output activations before pose regression
                        together with regressed poses, otherwise only poses.
    Return:
        xyz: global camera position.
        wpqr: global camera orientation in quaternion.
    """
    def __init__(self, regid, with_embedding=False):
        super(Regression, self).__init__()
        conv_in = {"regress1": 512, "regress2": 528}
        self.with_embedding = with_embedding
        if regid != "regress3":
            self.projection = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3),
                                            nn.Conv2d(conv_in[regid], 128, kernel_size=1),
                                            nn.ReLU())
            self.regress_fc_pose = nn.Sequential(nn.Linear(2048, 1024),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.7))
            self.regress_fc_xyz = nn.Linear(1024, 3)
            self.regress_fc_wpqr = nn.Linear(1024, 4)
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.regress_fc_pose = nn.Sequential(nn.Linear(1024, 2048),
                                                 nn.ReLU(),
                                                 nn.Dropout(0.5))
            self.regress_fc_xyz = nn.Linear(2048, 3)
            self.regress_fc_wpqr = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.projection(x)
        x = self.regress_fc_pose(x.view(x.size(0), -1))
        xyz = self.regress_fc_xyz(x)
        wpqr = self.regress_fc_wpqr(x)
        wpqr = F.normalize(wpqr, p=2, dim=1)
        if self.with_embedding:
            return (xyz, wpqr, x)
        return (xyz, wpqr)

class PoseNet_resnet(BaseNet):
    '''PoseNet model in [Kendall2015ICCV] Posenet: A convolutional network for real-time 6-dof camera relocalization.'''

    def __init__(self, config, with_embedding=False):
        super(PoseNet_resnet, self).__init__(config)
        self.extract = resnet34(pretrained=False)
        self.regress1 = Regression('regress1')
        self.regress2 = Regression('regress2')
        self.regress3 = Regression('regress3', with_embedding=with_embedding)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Loss params
        self.learn_weighting = config.learn_weighting
        if self.learn_weighting:
            # Learned loss weighting during training
            sx, sq = config.homo_init
            # Variances variables to learn
            self.sx = nn.Parameter(torch.tensor(sx))
            self.sq = nn.Parameter(torch.tensor(sq))
        else:
            # Fixed loss weighting with beta
            self.beta = config.beta

        self.to(self.device)
        self.init_weights_(config.weights_dict)
        self.set_optimizer_(config)

    def forward(self, x):
        if self.training:
            feat4a, feat4d, feat5b = self.extract(x)
            pose = [self.regress1(feat4a), self.regress2(feat4d), self.regress3(feat5b)]
        else:
            feat5b = self.extract(x)
            pose = self.regress3(feat5b)
        return pose

    def get_inputs_(self, batch, with_label=True):
        im = batch['im']
        im = im.to(self.device)
        if with_label:
            xyz = batch['xyz'].to(self.device)
            wpqr = batch['wpqr'].to(self.device)
            return im, xyz, wpqr
        else:
            return im

    def predict_(self, batch):
        pose = self.forward(self.get_inputs_(batch, with_label=False))
        xyz, wpqr = pose[0], pose[1]
        return xyz.data.cpu().numpy(), wpqr.data.cpu().numpy()

    def init_weights_(self, weights_dict):
        '''Define how to initialize the model'''

        if weights_dict is None:
            print('Initialize all weigths')
            self.apply(self.xavier_init_func_)
        elif len(weights_dict.items()) == len(self.state_dict()):
            print('Load all weigths')
            self.load_state_dict(weights_dict)
        else:
            print('Init only part of weights')
            self.apply(self.normal_init_func_)
            self.load_state_dict(weights_dict, strict=False)

    def loss_(self, batch):
        im, xyz_, wpqr_ = self.get_inputs_(batch, with_label=True)
        criterion = nn.MSELoss()
        pred = self.forward(im)
        loss = 0
        losses = []
        loss_weighting = [0.3, 0.3, 1.0]
        if self.learn_weighting:
            loss_func = lambda loss_xyz, loss_wpqr: self.learned_weighting_loss(loss_xyz, loss_wpqr, self.sx, self.sq)
        else:
            loss_func = lambda loss_xyz, loss_wpqr: self.fixed_weighting_loss(loss_xyz, loss_wpqr*3, beta=self.beta)
        for l, w in enumerate(loss_weighting):
            xyz, wpqr = pred[l]
            loss_xyz = criterion(xyz, xyz_)
            loss_wpqr = criterion(wpqr, wpqr_)
            losses.append((loss_xyz, loss_wpqr))  # Remove if not necessary
            loss += w * loss_func(loss_xyz, loss_wpqr)
        return loss, losses

    def ss_loss_(self, batch, log):
        def show(img):
            npimg = img.detach().numpy().reshape([3,224,224])
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
            plt.show()

        class MyDataset(Dataset):
            def __init__(self, data, transform):

                self.data = data

                # define your transform pipline
                self.transform = transform

            def __getitem__(self, index):
                x = self.data[index]
                return self.transform(x)

            def __len__(self):
                return self.data.shape[0]

        transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.Resize((224,224)),
             transforms.RandomCrop((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        im = self.get_inputs_(batch, with_label=False)
        im_dataset=MyDataset(im,transform=transform)
        im_dataset_loader = torch.utils.data.DataLoader(im_dataset, batch_size=im_dataset.__len__())
        criterion=nn.MSELoss(reduction="mean")

        for real_im in im_dataset_loader:
            real_im=real_im.to(self.device)
            pred = self.forward(real_im)
            losses = []
            loss_weighting = [0.3, 0.3, 1.0]
            loss_func = lambda loss_xyz : self.learned_weighting_loss_ss(loss_xyz)
            loss=0
            normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            for l, w in enumerate(loss_weighting):
                xyz, wpqr = pred[l]
                fake=evaluate_images(xyz, wpqr)
                fake_dataset=MyDataset(fake,transform=normalize)
                fake_dataset_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=fake_dataset.__len__())

                for fake in fake_dataset_loader:
                    fake=fake.to(self.device)
                    loss_f = criterion(fake,real_im)*10*0.5
                    """
                    while True:
                        loss_f = criterion(fake,real_im)*10*0.5
                        if(loss_f==loss_f):
                            break
                        fake=torch.round(fake)
                        real_im=torch.round(real_im)"""
                    loss+=w*loss_func(loss_f)
                    losses.append(loss)

        #fake1,fake2,fake3=cuda(fake)
        """
        loss_f1 = criterion(fake1,real_im) * 10 * 0.5
        loss_f2 = criterion(fake2,real_im) * 10 * 0.5
        loss_f3 = criterion(fake3,real_im) * 10 * 0.5
        lprint("loss_f1: {}, loss_f2: {}, loss_f3: {}".format(loss_f1,loss_f2, loss_f3),log)
        loss=loss_weighting[0]*torch.mean(loss_func(loss_f1))+loss_weighting[1]*torch.mean(loss_func(loss_f2))+loss_weighting[2]*torch.mean(loss_func(loss_f3))
        lprint("loss: {}, fake: {}, real: {}".format(loss, [fake1[0,:,1:3,1:3],fake2[0,:,1:3,1:3],fake3[0,:,1:3,1:3]], real_im[0,:,1:3,1:3]),log)
        losses.append(loss)"""
        return loss, losses