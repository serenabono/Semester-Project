import torch
import torch.nn as nn
from networks.utils import cuda
from collections import OrderedDict
import itertools
from torch.cuda.amp import GradScaler, autocast
from utils.common.setup import lprint


class BaseNet(nn.Module):
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = config
        self.device = config.device

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])


    def forward(self, *inputs):
        """Defines the computation performed at every call.
        Inherited from Superclass torch.nn.Module.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def get_inputs_(self, batch, **kwargs):
        """Define how to parse the data batch for the network training/testing.
        This allows the flexibility for different training data formats in different applications.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def init_weights_(self):
        """Define how to initialize the weights of the network.
        Should be overridden by all subclasses, since it 
        normally differs according to the network models.
        """
        raise NotImplementedError

    def ss_loss_(self, batch):
        """Define how to calculate the loss  for the network.
        Should be overridden by all subclasses, since different
        applications or network models may have different types
        of targets and the corresponding criterions to evaluate
        the predictions.
        """
        raise NotImplementedError

    def loss_(self, batch):
        """Define how to calculate the loss  for the network.
        Should be overridden by all subclasses, since different 
        applications or network models may have different types 
        of targets and the corresponding criterions to evaluate 
        the predictions.
        """
        raise NotImplementedError

    def set_optimizer_(self, config):
        if config.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr_init, eps=config.epsilon, weight_decay=config.weight_decay)
        elif config.optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=False)

        #self.g_optimizer = torch.optim.Adam(self.parameters(), lr=.0002, betas=(0.5, 0.999))

    # Initialize optimizer from a state if available
        if config.optimizer_dict and config.training:
            self.optimizer.load_state_dict(config.optimizer_dict)

        class LambdaLR():
            def __init__(self, epochs, offset, decay_epoch):
                self.epochs = epochs
                self.offset = offset
                self.decay_epoch = decay_epoch

            def step(self, epoch):
                return 1.0 - max(0, epoch + self.offset - self.decay_epoch) / (self.epochs - self.decay_epoch)

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=LambdaLR(1000, 0, 100).step)
        self.scaler=GradScaler()

        if config.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_factor, last_epoch=config.start_epoch-1)
        else:
            self.lr_scheduler = None

    def predict_(self, batch):
        return self.forward(*self.get_inputs_(batch, with_label=False))

    def optim_step_g_(self, batch):
        self.g_optimizer.zero_grad()
        loss, losses = self.ss_loss_(batch)
        loss.backward()
        self.g_optimizer.step()
        self.g_lr_scheduler.step()
        return loss, losses

    def optim_step_(self, batch,log,ss=False):
        self.optimizer.zero_grad()
        if(ss==False):
            loss, losses = self.loss_(batch)
            #lprint("loss_1: {}".format(loss),log)
        else:
            loss, losses = self.ss_loss_(batch, log)
            #lprint("loss_2: {}".format(loss),log)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        """loss.backward()
        self.optimizer.step()"""
        return loss, losses

    def train_epoch(self, data_loader, ss_data_loader,epoch,log):
        if self.lr_scheduler:
            self.lr_scheduler.step()
        for i, batch in enumerate(data_loader):
            loss, losses = self.optim_step_(batch,log,ss=False)
        if(epoch%5==0):
            if self.lr_scheduler:
                self.lr_scheduler.step()
            for i, batch in enumerate(ss_data_loader):
                loss_ss, losses_ss = self.optim_step_(batch,log,ss=True)
            return loss, losses, loss_ss,losses_ss
        return  loss, losses
    
    def save_weights_(self, sav_path):
        torch.save(self.state_dict(), sav_path)

    def print_(self):
        for k, v in self.state_dict().items():
            print(k, v.size())

    def kaiming_normal_init_func_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:  # Incase bias is turned off
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    def xavier_init_func_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def normal_init_func_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def fixed_weighting_loss(self, loss_pos, loss_rot, beta=1.0):
        '''The weighted loss function with beta as a hyperparameter to balance the positional loss and the rotational loss'''
        return loss_pos + beta * loss_rot

    def learned_weighting_loss(self, loss_pos, loss_rot, sx, sq):
        '''The weighted loss function that learns variables sx and sy to balance the positional loss and the rotational loss'''
        return (-1 * sx).exp() * loss_pos + sx + (-1 * sq).exp() * loss_rot + sq

    def learned_weighting_loss_ss(self,loss):
        return loss