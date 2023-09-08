import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import os
from function import RandomDataset, A, A_CDP, Poisson_noise_torch, psnr, ssim
from model import Ista_Prdeep
import random
from argparse import ArgumentParser
import time
import cv2
import glob
# ==============================================================================================================================
parser = ArgumentParser(description='Ista_Prdeep_Net')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200,
                    help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=7,
                    help='phase number of DPUNet')
parser.add_argument('--learning_rate', type=float,
                    default=1e-3, help='learning rate')
parser.add_argument('--group_num', type=int, default=1,
                    help='group number for training')
parser.add_argument('--matrix_dir', type=str,
                    default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--SamplingRate', type=int,
                    default=4, help='Sampling Rate')
parser.add_argument('--measurement_type', type=str,
                    default='CDP', help='forward model type')
parser.add_argument('--ResF', default=1, type=int,
                    help='whether use ResFBlock')
parser.add_argument('--atten', default=1, type=int, help='whether use CBAM')
parser.add_argument('--shared_ResF', default=0, type=int,
                    help='whether share the ResFBlock parameter among stages')
parser.add_argument('--shared_CBAM', default=0, type=int,
                    help='whether share CBAM parameter among stages')
parser.add_argument('--log_set', default=0, type=int,
                    help='whether use log--based loss function')
parser.add_argument('--imsize', type=int, default=128,
                    help='the size of test image')
parser.add_argument('--alpha', type=int, default=9, help='noise level')

args = parser.parse_args()
# ==============================================================================================================================
measurement_type = args.measurement_type  # 'Fourier','CDP'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Layernum = args.layer_num
model_name = 'PRISTA'
learning_rate = args.learning_rate
log_set = args.log_set
shared_ResF = args.shared_ResF
shared_CBAM = args.shared_CBAM
atten = args.atten
ResF = args.ResF
imsize = args.imsize
alpha = args.alpha
model = Ista_Prdeep(Layernum, measurement_type=measurement_type, atten=atten,
                    ResF=ResF, shared_ResF=shared_ResF, shared_CBAM=shared_CBAM)

model.to(device)

model_dir = "./model/%s_%s_Group_%d_layer_num_%d_lr_%.4f" % (
    model_name, measurement_type, args.group_num, Layernum, learning_rate)
log_file_name = "./%s/%s_Group_%d_layer_num_%d_lr_%.4f.txt" % (
    model_dir, model_name, args.group_num, Layernum, learning_rate)
start_epoch, end_epoch = args.start_epoch, args.end_epoch
model.load_state_dict(torch.load('%s/net_params_%d.pkl' %
                                 (model_dir, end_epoch)))
# ==============================================================================================================================
SamplingRate = args.SamplingRate
psnrs, ssims, outputs, times, imgs = [], [], [], [], []
files = glob.glob('./data/Test/TestImages_%d/*.png' % imsize)
for i in range(len(files)):
    print(files[i])
    img = cv2.imread(files[i], 0)/255
    img = img.astype(np.float32)
    imgs.append(img)
    h, w = img.shape
    img = torch.from_numpy(img).to(device).reshape(1, 1, h, w)
    Mask_data_Name = './%s/mask_%d_%d_test.p' % (
        'sampling_matrix', SamplingRate, h)
    if os.path.exists(Mask_data_Name):
        Mask_data = pickle.load(open(Mask_data_Name, 'rb'))
    else:
        Mask_data = torch.exp(
            1j*2*torch.pi*torch.rand(1, SamplingRate, h, w)).to(device)
        pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
    mask = Mask_data.to(device)
    b = Poisson_noise_torch(A_CDP(img, SamplingRate=SamplingRate,
                            mask=mask, device=device), alpha=alpha, device=device)
    init_data = torch.ones((1, 1, h, w), dtype=torch.float).to(device)
    start = time.time()
    with torch.no_grad():
        [x_output, predlosses, intermediate_imgs, pred_imgs] = model(
            init_data, b, img, SamplingRate, mask)
    end = time.time()
    times.append(round(end-start, 3))
    psnrs.append(round(psnr(x_output.detach().cpu().numpy()[
                 0][0]*255, img.detach().cpu().numpy()[0][0]*255), 3))
    ssims.append(round(ssim(x_output.detach().cpu().numpy()[
                 0][0]*255, img.detach().cpu().numpy()[0][0]*255), 4))
    outputs.append(x_output.detach().cpu().numpy()[0][0])

UNt_psnr = psnrs[:6]
NT_psnr = psnrs[6:]
UNt_ssim = ssims[:6]
NT_ssim = ssims[6:]
print('UNT_PSNR:', UNt_psnr, 'NT_PSNR:', NT_psnr)
print('AVERAGE UNT_PSNR:', round(np.mean(UNt_psnr), 3),
      'AVERAGE NT_PSNR:', round(np.mean(NT_psnr), 3))
print('UNT_SSIM:', UNt_ssim, 'NT_SSIM:', NT_ssim)
print('AVERAGE UNT_SSIM:', round(np.mean(UNt_ssim), 3),
      'AVERAGE NT_SSIM:', round(np.mean(NT_ssim), 3))
print('AVERAGE UNT_TIME:', round(
    np.mean(times[:6]), 3), 'AVERAGE NT_TIME:', round(np.mean(times[6:]), 3))
