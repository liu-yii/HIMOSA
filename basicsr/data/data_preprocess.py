import numpy as np
import torch
from glob import glob
import os
from PIL import Image
import shutil
import random


def preprocess_skymap():
### process SkyMap dataset

    data_dir = "F:/datasets/skymap"
    output_dir = "F:/datasets/skymap-SR"
    citys = ["Chicago", "Johannesburg", "London", "Rio", "Sydney", "Taipei", "Tokyo"]

    for city in citys:
        print(f"Processing {city} ...")
        city_dir = os.path.join(data_dir, city, "sat_img")
        all_dirs = os.listdir(city_dir)
        train_dirs = random.sample(all_dirs, min(2000, len(all_dirs)))

        # 选取不在sampled_dirs中的20个作为验证集
        remaining_dirs = list(set(all_dirs) - set(train_dirs))
        val_dirs = random.sample(remaining_dirs, min(20, len(remaining_dirs)))

        for idx, data in enumerate(train_dirs):
            data_path = os.path.join(city_dir, data, "satellite.jpg")
            output_path = os.path.join(output_dir, "train/hr", f"{city}_{idx:04d}.jpg")
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            shutil.copyfile(data_path, output_path)
        
        # 保存验证集
        for idx, data in enumerate(val_dirs):
            data_path = os.path.join(city_dir, data, "satellite.jpg")
            output_path = os.path.join(output_dir, "val/hr", f"{city}_{idx:04d}.jpg")
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            shutil.copyfile(data_path, output_path)


def preprocess_naip():
    ## Process NAIP dataset
    data_dir = "F:/datasets/naip_small"
    output_dir = "F:/datasets/naip-SR"

    all_paths = {}
    for i in os.listdir(data_dir):
        timestamp = i.split("_")[-1][:4]
        dir = os.path.join(data_dir, i, "tci")
        datas = [os.path.join(dir, item) for item in os.listdir(dir) if "_55639" not in item and "_55638" not in item]
        if timestamp not in all_paths:
            all_paths[timestamp] = []
        all_paths[timestamp].extend(datas)
    
    train_paths = []
    val_paths = []
    for timestamp, paths in all_paths.items():
        sampled_paths = random.sample(paths, min(200000, len(paths)))
        train_paths.extend(sampled_paths)
        # 选取不在sampled_dirs中的20个作为验证集
        remaining_paths = list(set(paths) - set(sampled_paths))
        val_paths.extend(random.sample(remaining_paths, min(200, len(remaining_paths))))
    print("Number of training images:", len(train_paths))

    for idx, data in enumerate(train_paths):
        output_path = os.path.join(output_dir, "train/hr", f"{idx:07d}.jpg")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        shutil.copyfile(data, output_path)

    # 保存验证集
    for idx, data in enumerate(val_paths):
        output_path = os.path.join(output_dir, "val/hr", f"{idx:07d}.jpg")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        shutil.copyfile(data, output_path)

def downsample(dir, scale = 4):
    hr_dir = os.path.join(dir, "hr")
    lr_dir = os.path.join(dir, f"lr_x{scale}")
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
    for img_name in os.listdir(hr_dir):
        hr_path = os.path.join(hr_dir, img_name)
        lr_path = os.path.join(lr_dir, img_name)
        img = Image.open(hr_path)
        w, h = img.size
        img_lr = img.resize((w // scale, h // scale), Image.BICUBIC)
        img_lr.save(lr_path)


if __name__ == "__main__":
    # preprocess_skymap()
    preprocess_naip()
    print("Start downsampling ...")
    downsample("F:/datasets/naip-SR/train", scale=2)
    downsample("F:/datasets/naip-SR/val", scale=2)   
    # downsample("/media/yi/F/datasets/skymap-SR/train", scale=2)
    # downsample("/media/yi/F/datasets/skymap-SR/val", scale=2)
    # downsample("/mnt/E/datasets/RSISR/AID_SR/val", scale=2)
    # downsample("/mnt/E/datasets/RSISR/AID_SR/train", scale=2)

        

    
