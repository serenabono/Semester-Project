import os
from argparse import ArgumentParser
import model as md
import test as tst
from utils import create_link



 #self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
 #self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, 
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
 #self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
 #self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--fineSize', type=int, default=256)
    parser.add_argument('--loadSize', type=int, default=256)
    parser.add_argument('--pool_size', type=int, default=5)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=256)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=256)
    parser.add_argument('--crop_width', type=int, default=256)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='./results_robocup_wrobot-yellow_batch=1_no_weights/')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/robocup-thick/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/robocup-thick/')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    args = parser.parse_args()
    return args


def main():
  args = get_args()
  create_link(args.dataset_dir)
  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []
  for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
      args.gpu_ids.append(id)
  print(not args.no_dropout)
  if args.training:
      print("Training")
      model = md.cycleGAN(args)
      model.train(args)
  if args.testing:
      print("Testing")
      tst.test(args)


#if __name__ == '__main__':
main()
