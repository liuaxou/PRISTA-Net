import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch._jit_internal import Optional
from function import A, AT, A_CDP, At_CDP

# Define initialize parametes


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True, transpose=False,
                 channel_shuffle_g=0, norm_method=nn.BatchNorm2d, groups=1):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias, groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBLOCK(nn.Module):
    def __init__(self, n_feat):
        super(ResBLOCK, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        x_res = self.main(x)
        out = x+x_res
        return F.relu(out)


class ResFBLOCK(nn.Module):
    def __init__(self, n_feat, norm='backward'):  # 'ortho'
        super(ResFBLOCK, self).__init__()
        self.main = nn.Sequential(
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=True),
            BasicConv(n_feat, n_feat, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = nn.Sequential(
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=True),
            BasicConv(n_feat*2, n_feat*2, kernel_size=1, stride=1, relu=False)
        )
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y

# ============================================================================
# PrdeepIsta
# ============================================================================
# CBAM


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)*x
        x = self.sa(x)*x
        return x


class Basicblock_Attention(nn.Module):
    def __init__(self, measurement_type, RESFBlock1, RESFBLOCK2, CBAM1, CBAM2, atten, ResF, shared_ResF, shared_CBAM, back_channel=32):
        super(Basicblock_Attention, self).__init__()

        self.tau = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.D = nn.Conv2d(1, 32, 3, padding=1)
        if ResF == False:
            print('ResF==False')
            self.RESF1 = None
            self.RESF2 = None
        elif shared_ResF == True and ResF == True:
            print('ResF==True')
            self.RESF1 = RESFBlock1
            self.RESF2 = RESFBLOCK2
        elif shared_ResF == False and ResF == True:
            print('ResF==True')
            self.RESF1 = ResFBLOCK(32)
            self.RESF2 = ResFBLOCK(back_channel)

        self.layer_forward = nn.Sequential(
            *[BasicConv(32, 32, 3, 1, True) for i in range(3)])

        self.layer_backward = nn.Sequential(
            *[BasicConv(32, 32, 3, 1, True) for i in range(3)])
        if atten == False:
            print('atten==False')
            self.CBAM1 = None
            self.CBAM2 = None
        elif shared_CBAM == True and atten == True:
            print('atten==True')
            self.CBAM1 = CBAM1
            self.CBAM2 = CBAM2
        elif shared_CBAM == False and atten == True:
            print('atten==True')
            self.CBAM1 = CBAM(32, 3)
            self.CBAM2 = CBAM(32, 3)

        self.G = nn.Conv2d(32, 1, 3, padding=1)

        self.measurement_type = measurement_type

        if self.measurement_type == 'Fourier':
            self.A = A
            self.AT = AT
        else:
            self.A = A_CDP
            self.AT = At_CDP

    def forward(self, x, b, original_data, SamplingRate, mask):
        '''
        x: initial data shape: (batch, 1, imsize1, imsize2)
        b: measurement data 
        original_data: shape: (batch, 1, imsize1, imsize2)
        '''
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # subgradient descent module to close true vaule
        if self.measurement_type == 'Fourier':
            z = self.A(x)
            x_input = x - self.tau * self.AT(z-b*(z/(torch.abs(z)+1e-8)))
        else:
            z = self.A(x, SamplingRate, mask=mask, device=device)
            x_input = x - self.tau * \
                self.AT(z-b*(z/(torch.abs(z)+1e-8)),
                        SamplingRate=SamplingRate, mask=mask)
        # proximal-point mapping module to denoise
        x_forward = self.D(x_input)
        if self.CBAM1 != None:
            x_forward = self.CBAM1(x_forward)
        x_forward = self.layer_forward(x_forward)
        if self.RESF1 != None:
            x_forward = self.RESF1(x_forward)

        x_1 = x_forward
        # soft thresholding to sparse feature
        x_backward = torch.mul(torch.sign(x_1), F.relu(
            torch.abs(x_1) - self.soft_thr))
        if self.RESF2 != None:
            x_backward = self.RESF2(x_backward)
            x_1 = self.RESF2(x_1)
        x_backward = self.layer_backward(x_backward)
        x_1 = self.layer_backward(x_1)
        if self.CBAM2 != None:
            x_backward = self.CBAM2(x_backward)
            x_1 = self.CBAM2(x_1)
        x_G = self.G(x_backward)
        x_1 = self.G(x_1)
        x_pred = x_G+x_input

        predloss = x_pred - original_data
        symloss = x_1 - x_input
        return [x_input, x_pred, predloss, symloss]


class Ista_Prdeep(nn.Module):
    def __init__(self, Layerno, measurement_type, atten=True, ResF=True, shared_ResF=False, shared_CBAM=False):
        super(Ista_Prdeep, self).__init__()
        self.Layerno = Layerno  # iterate steps
        onelayer = []
        if shared_ResF == True:
            self.RESF1 = ResFBLOCK(32)
            self.RESF2 = ResFBLOCK(32)
        else:
            self.RESF1 = None
            self.RESF2 = None

        if shared_CBAM == True:
            self.CBAM1 = CBAM(32, 3)
            self.CBAM2 = CBAM(32, 3)
        else:
            self.CBAM1 = None
            self.CBAM2 = None

        for i in range(Layerno):
            onelayer.append(Basicblock_Attention(measurement_type, atten=atten, ResF=ResF, CBAM1=self.CBAM1, CBAM2=self.CBAM2,
                                                 RESFBlock1=self.RESF1, RESFBLOCK2=self.RESF2, shared_ResF=shared_ResF, shared_CBAM=shared_CBAM))
        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

    def forward(self, x, b, original_data, SamplingRate, mask=None):
        '''
        x:(batch,1,imsize1,imsize2)
        b:(batch,1,imsize1*2,imsize2*2)
        original_data:(batch,1,imsize1,imsize2)
        '''
        predlosses = []
        intermediate_imgs = []
        pred_imgs = []
        symlosses = []
        for i in range(self.Layerno):
            [r, x, predloss, symloss] = self.fcs[i](
                x, b, original_data, SamplingRate, mask)
            predlosses.append(predloss)
            intermediate_imgs.append(r)
            pred_imgs.append(x)
            symlosses.append(symloss)
        x_final = x

        return [x_final, predlosses, intermediate_imgs, pred_imgs, symlosses]
# =============================================================================


def model_structure(model):
    '''
    caculate the number of model parameters
    '''
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|'
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

