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
import cv2


def main(root_path, imgpath, w=128, h=128, window_size=32, fold=50, sigma=1.2, l=9, alpha=0.3, zoomfactor=4, kde=False):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True
    
    # create model
    model = build_model(opt)
    net = model.net_g

    # 准备图像
    # Prepare images
    img_lr, img_hr = prepare_images(imgpath, scale=zoomfactor)
    tensor_lr = PIL2Tensor(img_lr)[:3]
    tensor_hr = PIL2Tensor(img_hr)[:3]


    w, h, position_pil = click_select_position(img_hr, window_size)

    attr_objective = attribution_objective(attr_grad, h, w, window=window_size)
    gaus_blur_path_func = GaussianBlurPath(sigma, fold, l)
    interpolated_grad_numpy, result_numpy, interpolated_numpy = Path_gradient(tensor_lr.numpy(), net, attr_objective,
                                                                            gaus_blur_path_func, cuda=True)

    grad_numpy, result = saliency_map(interpolated_grad_numpy, result_numpy)
    abs_normed_grad_numpy = grad_abs_norm(grad_numpy)
    saliency_image_abs = vis_saliency(abs_normed_grad_numpy, zoomin=4)
    saliency_image_kde = vis_saliency_kde(abs_normed_grad_numpy)
    blend_abs_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    blend_kde_and_input = cv2_to_pil(
        pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + pil_to_cv2(img_lr.resize(img_hr.size)) * alpha)
    pil = make_pil_grid(
        [position_pil,
        saliency_image_abs,
        blend_abs_and_input,
        blend_kde_and_input
        ]
    )
    # pil.show()
    result_dir = "visual/" 
    os.makedirs(result_dir, exist_ok=True)
    pil.save(result_dir + opt['name'] + ".png")

    gini_index = gini(abs_normed_grad_numpy)
    diffusion_index = (1 - gini_index) * 100
    print(f"The DI is {diffusion_index}")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path, imgpath="demo/AID/P0798.png")