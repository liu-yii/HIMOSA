import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile
from tqdm import tqdm
import numpy as np

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0*g:1*g, 1, 2] = 1.0
        mask[:, 1*g:2*g, 1, 0] = 1.0
        mask[:, 2*g:3*g, 2, 1] = 1.0
        mask[:, 3*g:4*g, 0, 1] = 1.0
        mask[:, 4*g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1) 
        return y


class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0*g:1*g, 0, 1, 2] = 1.0 ## left
        self.weight[1*g:2*g, 0, 1, 0] = 1.0 ## right
        self.weight[2*g:3*g, 0, 2, 1] = 1.0 ## up
        self.weight[3*g:4*g, 0, 0, 1] = 1.0 ## down
        self.weight[4*g:, 0, 1, 1] = 1.0 ## identity     

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y) 
        return y


class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()    
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory': 
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y

class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='relu'):
        super(LFE, self).__init__()    
        self.exp_ratio = exp_ratio
        self.act_type  = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels*exp_ratio)
        self.conv1 = ShiftConv2d(out_channels*exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.conv1(y) 
        return y

class GMSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[4, 8, 12], calc_attn=True):
        super(GMSA, self).__init__()    
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns  = [channels*2//3, channels*2//3, channels*2//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels*2, kernel_size=1), 
                nn.BatchNorm2d(self.channels*2)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns  = [channels//3, channels//3,channels//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1), 
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', 
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1)) 
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c', 
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, prev_atns

class ELAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12], shared_depth=1):
        super(ELAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth
        
        modules_lfe = {}
        modules_gmsa = {}
        modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_gmsa['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            modules_lfe['lfe_{}'.format(i+1)] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
            modules_gmsa['gmsa_{}'.format(i+1)] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=False)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)

    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, None)
                x = y + x
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, atn = self.modules_gmsa['gmsa_{}'.format(i)](x, atn)
                x = y + x
        return x
    
class ELAN(nn.Module):
    def __init__(self, args):
        super(ELAN, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.m_elan  = args.m_elan
        self.c_elan  = args.c_elan
        self.n_share = args.n_share
        self.r_expand = args.r_expand
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)

        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ELAN')
    parser.add_argument('--scale', type=int, default=4, help='super resolution scale')
    parser.add_argument('--colors', type=int, default=3, help='number of color channels to use')
    parser.add_argument('--window_sizes', type=int, nargs='+', default=[4, 8, 16],
                        help='window sizes for GMSA')
    parser.add_argument('--m_elan', type=int, default=36, help='number of ELAB blocks in ELAN')
    parser.add_argument('--c_elan', type=int, default=180, help='number of channels in ELAN')
    parser.add_argument('--n_share', type=int, default=0, help='number of shared layers in ELAB')
    parser.add_argument('--r_expand', type=int, default=2, help='expansion ratio in LFE')
    parser.add_argument('--rgb_range', type=int, default=255, help='rgb range')
    args = parser.parse_args()  
    model = ELAN(args).cuda()
    # print(model)
    # print(height, width, model.flops() / 1e9)

    # x = torch.randn((1, 3, height, width))
    # x = model(x)
    # print(x.shape)
    input = torch.rand(1, 3, 256, 256).cuda()  # B C H W
    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    flops, _ = profile(model, inputs=(input,))
    print("FLOPs: {} G".format(flops/1e9))

    def cal_time(net, cal_iter=50):
        img0 = torch.randn(size=(1, 3, 256, 256)).float().cuda()
        # warmup
        with torch.no_grad():
            for i in range(10):
                _ = net(img0)
            torch.cuda.synchronize()

        # 设置用于测量时间的 cuda Event
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 初始化一个时间容器
        timings = np.zeros((cal_iter, 1))
        pbar = tqdm(total=cal_iter)
        with torch.no_grad():
            for rep in range(cal_iter):
                starter.record()
                _ = net(img0)
                ender.record()
                torch.cuda.synchronize()  # 等待GPU任务完成
                curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
                timings[rep] = curr_time
                pbar.update()
                pbar.set_description("testing: " + str(curr_time))
        pbar.close()
        avgtime = timings.sum() / cal_iter

        return avgtime


    time_cost = cal_time(model)
    print('\033[0;36m time {} \033[0m'.format(time_cost))