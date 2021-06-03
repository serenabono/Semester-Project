import itertools
import functools

import numpy as np
import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis, set_grad
import tensorflow as tf
from test import test
'''
Class for CycleGAN with train() as a member function

'''
class cycleGAN(object):
    def __init__(self,args):

        # Define the network 
        #####################################################
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=2, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=2, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])

        # Define Loss criterias

        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))
        

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self,args):
        """
        def get_weights(image):

            C,W,H = image[0].shape  # this returns batch_size, 128, 128, 5
            cmap_flat = tf.reshape(image[0], shape=[C,W*H])
            cmap_flat=((cmap_flat+1)/2)*255
            cmap_flat=tf.transpose(cmap_flat)
            idx_green=np.intersect1d(np.argwhere(np.sum(cmap_flat,axis=1)<200), (np.argwhere(np.sum(cmap_flat,axis=1)>160)))
            idx_gray=np.intersect1d(np.argwhere(np.sum(cmap_flat,axis=1)<400), (np.argwhere(np.sum(cmap_flat,axis=1)>370)))
            idx_black=np.intersect1d(np.argwhere(np.sum(cmap_flat,axis=1)<10), (np.argwhere(np.sum(cmap_flat,axis=1)>-1)))
            idx_white=np.intersect1d(np.argwhere(np.sum(cmap_flat,axis=1)<220), (np.argwhere(np.sum(cmap_flat,axis=1)>200)))

            weights=(np.asarray([len(idx_green),len(idx_black),len(idx_gray),len(idx_white)])/(len(idx_green)+len(idx_white)+len(idx_black)+len(idx_gray)))

            colors=[idx_green.reshape([1,len(idx_green)]),idx_black.reshape([1,len(idx_black)]),idx_gray.reshape([1,len(idx_gray)]),idx_white.reshape([1,len(idx_white)])]
            return weights,colors
        """
        #clear cache
        torch.cuda.empty_cache()
        # For transforming the input image
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height,args.load_width)),
             transforms.RandomCrop((args.crop_height,args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Pytorch dataloader
        a_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainA'], transform=transform), 
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4,)
        b_loader = torch.utils.data.DataLoader(dsets.ImageFolder(dataset_dirs['trainB'], transform=transform),
                                                        batch_size=args.batch_size, shuffle=True, num_workers=4)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):
            if(epoch!=0):
                test(args,"#"+str(epoch)+",")
            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                # step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1
                torch.cuda.empty_cache()

                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                a_real = Variable(a_real[0])
                b_real = Variable(b_real[0])

                #weights_b_real,color_b_real=get_weights(b_real)

                a_real, b_real = utils.cuda([a_real, b_real])

                # Forward pass through generators
                ##################################################
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef

                """
                b_idt_loss=0
                C,W, H = b_real[0].shape  # this returns batch_size, 128, 128, 5
                cmap_flat0 = tf.transpose(tf.reshape(b_real[0], shape=[C,W*H]))
                
                C,W, H = b_idt[0].shape
                cmap_flat1 = tf.transpose(tf.reshape(Variable(b_idt[0]), shape=[C,W*H]))
                
                for idx in range(len(color_b_real)):
                    if(color_b_real[idx].shape[1]==0):
                        continue
                    b_real1= Variable(torch.tensor(tf.gather(cmap_flat0, torch.tensor(color_b_real[idx] ,dtype=torch.int64)).numpy()))
                    b_idt1= Variable(torch.tensor(tf.gather(cmap_flat1, torch.tensor(color_b_real[idx],dtype=torch.int64)).numpy()))
                    b_idt_loss += torch.tensor(weights_b_real[idx]*self.L1(b_idt1, b_real1)* args.lamda)
                    
                b_idt_loss=Variable(b_idt_loss.float().to("cuda"),requires_grad=True)
                print(b_idt_loss)
                
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef
                """
                # Adversarial losses
                ################################################### 
                
                a_fake_dis =self.Da(a_fake)
                b_fake_dis=self.Db(b_fake)

                real_labelb = utils.cuda(Variable(torch.ones(b_fake_dis.size())))
                real_labela = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                b_gen_loss=self.MSE(b_fake_dis, real_labelb)
                a_gen_loss=self.MSE(a_fake_dis, real_labela)
                """
                b_gen_loss = Variable(torch.tensor(self.MSE(b_fake_dis, real_labelb).item()*0.7).float().to("cuda"), requires_grad=True)
                a_gen_loss = Variable(torch.tensor(self.MSE(a_fake_dis, real_labela).item()*0.3).float().to("cuda"), requires_grad=True)
                """
                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda
                """
                b_cycle_loss=0
                C,W, H = b_real[0].shape  # this returns batch_size, 128, 128, 5
                cmap_flat0 = tf.transpose(tf.reshape(b_real[0], shape=[C,W*H]))

                C,W, H = b_idt[0].shape
                cmap_flat1 = tf.transpose(tf.reshape(Variable(b_recon[0]), shape=[C,W*H]))

                for idx in range(len(color_b_real)):
                    if(color_b_real[idx].shape[1]==0):
                        continue
                    b_real2= Variable(torch.tensor(tf.gather(cmap_flat0, torch.tensor(color_b_real[idx] ,dtype=torch.int64)).numpy()))
                    b_recon2= Variable(torch.tensor(tf.gather(cmap_flat1, torch.tensor(color_b_real[idx],dtype=torch.int64)).numpy()))
                    b_cycle_loss += torch.tensor(weights_b_real[idx]*self.L1(b_recon2, b_real2 * args.lamda))

                b_cycle_loss=Variable(b_cycle_loss.float().to("cuda"),requires_grad=True)
                """

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()


                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                #################################################
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_labela = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                real_labelb = utils.cuda(Variable(torch.ones(b_real_dis.size())))
                fake_labela = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))
                fake_labelb = utils.cuda(Variable(torch.zeros(b_fake_dis.size())))
                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_labela)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_labela)
                b_dis_real_loss = self.MSE(b_real_dis, real_labelb)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_labelb)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                                            (epoch, i + 1, min(len(a_loader), len(b_loader)),
                                                            gen_loss,a_dis_loss+b_dis_loss))

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()



