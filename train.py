import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import os
from function import RandomDataset, A, A_CDP, Poisson_noise_torch
from model import Ista_Prdeep
import random
from argparse import ArgumentParser

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
args = parser.parse_args()
# ==============================================================================================================================
# data load
measurement_type = args.measurement_type  # 'Fourier','CDP'
train_dataset = pickle.load(open('./data/Train/TrainImgs.p', 'rb'))
if measurement_type == 'Fourier':
    init_train_dataset = pickle.load(
        open('./data/Train/Pn-hioinit-TrainImgs.p', 'rb'))
else:
    init_train_dataset = np.ones_like(train_dataset)

batch_size = 10
train_loader = DataLoader(dataset=RandomDataset(
    train_dataset, len(train_dataset)), batch_size=batch_size, shuffle=False)
init_train_loader = DataLoader(dataset=RandomDataset(init_train_dataset, len(
    init_train_dataset)), batch_size=batch_size, shuffle=False)
# ==============================================================================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
Layernum = args.layer_num
model_name = 'Ista_Prdeep'
learning_rate = args.learning_rate
log_set = args.log_set
shared_ResF = args.shared_ResF
shared_CBAM = args.shared_CBAM
atten = args.atten
ResF = args.ResF
model = Ista_Prdeep(Layernum, measurement_type=measurement_type, atten=atten,
                    ResF=ResF, shared_ResF=shared_ResF, shared_CBAM=shared_CBAM)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.95)

model_dir = "./model/%s_%s_Group_%d_layer_num_%d_lr_%.4f" % (
    model_name, measurement_type, args.group_num, Layernum, learning_rate)
log_file_name = "./%s/%s_Group_%d_layer_num_%d_lr_%.4f.txt" % (
    model_dir, model_name, args.group_num, Layernum, learning_rate)
start_epoch, end_epoch = args.start_epoch, args.end_epoch
out_data_dir = './data/out/Group_%d_TrainPhase-%s-%s' % (
    args.group_num, model_name, measurement_type)
record_dir = './%s/%s-%s-Group_%d-loss.p' % (
    model_dir, model_name, measurement_type, args.group_num)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(out_data_dir):
    os.makedirs(out_data_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('%s/net_params_%d.pkl' %
                          (pre_model_dir, start_epoch)))
# ==============================================================================================================================
loss_record = []
SamplingRate = args.SamplingRate

for epoch_i in range(start_epoch+1, end_epoch+1):

    for i, data in enumerate(zip(train_loader, init_train_loader)):

        oringinal_data = data[0].to(device)
        # init data
        init_data = data[1].to(device)
        h, w = oringinal_data[0][0].shape
        if measurement_type == 'Fourier':
            alpha = random.choice([2, 3, 4])  # 'Fourier'=[2,3,4];‘CDP’=[9,27]
        else:
            alpha = random.choice([9, 27, 81])
        # caculate the magnitudes of batch data and add possio noise
        if measurement_type == 'Fourier':
            mask = None
            b = Poisson_noise_torch(
                A(oringinal_data), alpha=alpha, device=device)
        else:
            Mask_data_Name = './%s/mask_%d_%d_train.p' % (
                args.matrix_dir, SamplingRate, h)
            if os.path.exists(Mask_data_Name):
                Mask_data = pickle.load(open(Mask_data_Name, 'rb'))
            else:
                Mask_data = torch.exp(
                    1j*2*torch.pi*torch.rand(1, SamplingRate, h, w)).to(device)
                pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
            mask = Mask_data.to(device)
            b = Poisson_noise_torch(A_CDP(oringinal_data, SamplingRate=SamplingRate,
                                    mask=mask, device=device), alpha=alpha, device=device)

        model.train()
        # transmit the initial data and measurement data to the model
        [x_output, predlosses, intermediate_imgs, pred_imgs] = model(
            init_data, b, oringinal_data, SamplingRate, mask)
        # loss between ground truth and output of each stage
        loss_pred_stage = torch.mean(torch.pow(predlosses[0], 2))
        for k in range(1, Layernum):
            loss_pred_stage += torch.mean(torch.pow(predlosses[k], 2))

        if log_set == False:
            loss_all = torch.log(loss_pred_stage)
        else:
            loss_all = loss_pred_stage

        # zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d-%d/%02d] Total Loss: %.6f, Discrepancy Loss: %.6f\n" % (
            epoch_i, i+1, end_epoch, loss_all.item(), loss_pred_stage.item())
        print(output_data)
    scheduler.step()
    loss_record.append(loss_all.detach().cpu().numpy())
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)

    if epoch_i % 25 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" %
                   (model_dir, epoch_i))  # save only the parameters
        print("The learning rate of %d epoch: %.8f" %
              (epoch_i, optimizer.param_groups[0]['lr']))

pickle.dump(loss_record, open(record_dir, 'wb'))
