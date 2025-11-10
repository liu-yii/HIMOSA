import logging
import torch
from os import path as osp
import os
import numpy as np
import math

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.archs import build_network
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

from lam.utils import prepare_clips, vis_saliency, vis_saliency_kde, click_select_position, grad_abs_norm, \
    make_pil_grid, pil_to_cv2, cv2_to_pil, PIL2Tensor, Tensor2PIL, prepare_images, gini
from lam.core import attr_grad
from lam.BackProp import GaussianBlurPath
from lam.BackProp import attribution_objective, Path_gradient
from lam.BackProp import saliency_map_PG as saliency_map
from PIL import Image
from matplotlib import pyplot as plt
import cv2



def main(root_path, imgpath, input_H=128, input_W=128, zoomfactor=4, kde=False):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True
    
    # create model
    model = build_model(opt)
    net = model.net_g
    heatmap = np.zeros([input_H, input_W])
    #打印一下模型，选择其中的一个层
    print(net)

    #这里选择骨干网络的最后一个模块
    layer = net.layers[-1]
    print(layer)

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)


    # 准备图像
    # Prepare images
    img_lr, img_hr = prepare_images(imgpath, scale=zoomfactor)
    tensor_lr = PIL2Tensor(img_lr)[:3].unsqueeze(0).cuda()  # 添加batch维度
    tensor_hr = PIL2Tensor(img_hr)[:3].unsqueeze(0).cuda()  # 添加batch维度
    tensor_lr.requires_grad = True  # 需要计算梯度
    fmap_block = list()
    input_block = list()

    layer.register_forward_hook(forward_hook)
    # 前向传播
    output = net(tensor_lr)

    #特征图的channel维度算均值且去掉batch维度，得到二维张量
    if isinstance(fmap_block[0], tuple):
        feature_map = fmap_block[0][0]
    else:
        feature_map = fmap_block[0]
    feature_map = feature_map.transpose(1, 2).view(1, -1, input_H, input_W)
    # # 特征图在通道维度取均值，得到二维特征图
    # vis_feat = feature_map.mean(dim=1).cpu().detach().numpy()
    # vis_feat = np.clip(vis_feat, 0, None)  # 去掉负值
    # # vis_feat = (vis_feat - vis_feat.min()) / (vis_feat.max() - vis_feat.min() + 1e-8)  # 归一化，防止除0
    # plt.imshow(vis_feat[0], cmap='jet')
    # plt.colorbar()
    # plt.title('Feature Map Heatmap')
    # plt.savefig(f"visual/feature_map_heatmap_{opt['name']}.png")
    # plt.close()

    feature_map = feature_map.mean(dim=1,keepdim=False).squeeze()
    print("feature_map shape:", feature_map.shape)
    
    
    #对二维张量中心点（标量）进行backward
    feature_map[64,64].backward(retain_graph=True)

    #对输入层的梯度求绝对值
    grad = torch.abs(tensor_lr.grad)
    
    #梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
    grad = grad.mean(dim=1,keepdim=False).squeeze()
    
    #累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
    heatmap = heatmap + grad.cpu().numpy()
    
    cam = heatmap

    #对累加的梯度进行归一化
    cam = cam / cam.max()

    # 可视化，使用更易分辨的COLORMAP_TURBO
    cam = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    # # 将热力图叠加到原图上
    cam = cv2.resize(cam, (img_hr.size[0], img_hr.size[1]))
    img_hr = np.array(img_hr)
    cam = cv2.addWeighted(img_hr, 0.1, cam, 0.9, 0)
    cam = Image.fromarray(cam)
    cam.save(f"visual/ERF_{opt['name']}.png")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path, imgpath="demo/AID/denseresidential_370.png")