# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import time
import os
import scipy.io
import yaml
from PIL import Image
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
from utils import fuse_all_conv_bn
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')


WORKERS = 12


def main():

    ######################################################################
    # Options
    # --------

    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
    parser.add_argument('--test_dir',default='../Market/pytorch',type=str, help='./test_data')
    parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
    parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
    parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
    parser.add_argument('--PCB', action='store_true', help='use PCB' )
    parser.add_argument('--multi', action='store_true', help='use multiple query' )
    parser.add_argument('--fp16', action='store_true', help='use fp16.' )
    parser.add_argument('--ibn', action='store_true', help='use ibn.' )
    parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

    opt = parser.parse_args()
    ###load config###
    # load the training config
    config_path = os.path.join('./model',opt.name,'opts.yaml')
    with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'
    opt.fp16 = config['fp16']
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.use_NAS = config['use_NAS']
    opt.stride = config['stride']
    if 'use_swin' in config:
        opt.use_swin = config['use_swin']
    if 'use_swinv2' in config:
        opt.use_swinv2 = config['use_swinv2']
    if 'use_convnext' in config:
        opt.use_convnext = config['use_convnext']
    if 'use_efficient' in config:
        opt.use_efficient = config['use_efficient']
    if 'use_hr' in config:
        opt.use_hr = config['use_hr']

    if 'nclasses' in config: # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 751

    if 'ibn' in config:
        opt.ibn = config['ibn']
    if 'linear_num' in config:
        opt.linear_num = config['linear_num']

    str_ids = opt.gpu_ids.split(',')
    #which_epoch = opt.which_epoch
    name = opt.name
    test_dir = opt.test_dir

    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >=0:
            gpu_ids.append(id)

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    if opt.use_swin:
        h, w = 224, 224
    else:
        h, w = 256, 128

    data_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if opt.PCB:
        data_transforms = transforms.Compose([
            transforms.Resize((384,192), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        h, w = 384, 192

    data_dir = test_dir

    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery']}

    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load model
    #---------------------------
    def load_network(network):
        save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
                print("Compiling model...")
                # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
                torch.set_float32_matmul_precision('high')
                network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
            network.load_state_dict(torch.load(save_path))

        return network


    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from a trained model.
    #
    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip


    ######################################################################
    # Load Collected data Trained model
    if opt.use_dense:
        model_structure = ft_net_dense(opt.nclasses, stride = opt.stride, linear_num=opt.linear_num)
    elif opt.use_NAS:
        model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swin:
        model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_swinv2:
        model_structure = ft_net_swinv2(opt.nclasses, (h,w),  linear_num=opt.linear_num)
    elif opt.use_convnext:
        model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_efficient:
        model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
    elif opt.use_hr:
        model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
    else:
        model_structure = ft_net(opt.nclasses, stride = opt.stride, ibn = opt.ibn, linear_num=opt.linear_num)

    if opt.PCB:
        model_structure = PCB(opt.nclasses)

    print('=================')
    model = load_network(model_structure)
    print('=================')

    # Remove the final fc layer and classifier layer
    if opt.PCB:
        #if opt.fp16:
        #    model = PCB_test(model[1])
        #else:
            model = PCB_test(model)
    else:
        #if opt.fp16:
            #model[1].model.fc = nn.Sequential()
            #model[1].classifier = nn.Sequential()
        #else:
            model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    print('Here I fuse conv and bn for faster inference, and it does not work for transformers. Comment out this following line if you do not want to fuse conv&bn.')
    model = fuse_all_conv_bn(model)

    # We can optionally trace the forward method with PyTorch JIT so it runs faster.
    # To do so, we can call `.trace` on the reparamtrized module with dummy inputs
    # expected by the module.
    # Comment out this following line if you do not want to trace.
    #dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
    #model = torch.jit.trace(model, dummy_forward_input)

    # Extract feature
    with torch.no_grad():
        # with Image.open('../Market/pytorch/query/1103/1103_c6s3_011242_00.jpg') as image:
        #     img: torch.Tensor = data_transforms(image)
        with Image.open('test-image2.png').convert('RGB') as image:
            img: torch.Tensor = data_transforms(image)

        print(img.shape)
        img = img.unsqueeze(0)
        n, c, h, w = img.size()

        img = fliplr(img)
        input_img = Variable(img.cuda())
        ff: torch.Tensor = torch.FloatTensor(n, opt.linear_num).zero_().cuda()
        ff += model(input_img)
        # ---- L2-norm Feature ------
        ff = ff.data.cpu()
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    result = scipy.io.loadmat('pytorch_result.mat')
    gf = torch.FloatTensor(result['gallery_f'])

    # Sort images
    query = ff.view(-1, 1)
    print(query.shape)
    print(gf.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]

    for i in range(10):
        print(score[index[i]])
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        print(img_path)


if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time() - since
    print('Sort images in {:.5f}s'.format(time_elapsed % 60))
