import os
import sys
import shutil
import time
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *
import argparse

import torch
from engine import segmentation_engine
from utils import torch_dice_coef_loss

sys.setrecursionlimit(40000)


class Args:
    def __init__(self):
        self.data_set = "DRIVE"
        self.init = "barlowtwins"
        self.proxy_dir = "checkpoint.pth"
        self.train_data_dir = "DRIVE\\train\\images"
        self.train_mask_dir = "DRIVE\\train\\labels"
        self.valid_data_dir = "DRIVE\\validate\\images"
        self.valid_mask_dir = "DRIVE\\validate\\labels"
        self.test_data_dir = "DRIVE\\test\\images"
        self.test_mask_dir = "DRIVE\\test\\labels"
        self.train_batch_size = 32
        self.test_batch_size = 48
        self.epochs = 200
        self.train_num_workers = 2
        self.test_num_workers = 2
        self.distributed = False
        self.resume = ''
        self.learning_rate = 0.001
        self.mode = 'train'
        self.backbone = 'resnet50'
        self.arch = 'unet'
        self.device = "cuda"
        self.run = 1
        self.normalization = None
        self.activate = "sigmoid"
        self.patience = 20


def main(args):
    print(args)
    assert args.train_data_dir is not None
    assert args.data_set is not None
    assert args.train_mask_dir is not None
    assert args.valid_data_dir is not None
    assert args.valid_mask_dir is not None
    assert args.test_data_dir is not None
    assert args.test_mask_dir is not None

    if args.init.lower() != 'imagenet' and args.init.lower() != 'random':
        assert args.proxy_dir is not None

    if args.init is not None:
        model_path = os.path.join("./Models/Segmentation", args.data_set, args.arch, args.backbone, args.init,str(args.run))
    else:
        model_path = os.path.join("./Models/Segmentation", args.data_set, args.arch, args.backbone, "random",str(args.run))

    if args.data_set == "Montgomery":
        dataset_train = MontgomeryDataset(args.train_data_dir,args.train_mask_dir,transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = MontgomeryDataset(args.valid_data_dir,args.valid_mask_dir,transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = MontgomeryDataset(args.test_data_dir,args.test_mask_dir,transforms=None, normalization=args.normalization)
        criterion = torch_dice_coef_loss
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    if args.data_set == "DRIVE":
        dataset_train = DriveDataset(args.train_data_dir,args.train_mask_dir)
        dataset_val = DriveDataset(args.valid_data_dir,args.valid_mask_dir)
        dataset_test = DriveDataset(args.test_data_dir,args.test_mask_dir)
        criterion = torch.nn.BCELoss()
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

    if args.data_set == "SIIM_PNE": #Pneumothorax segmentation
        dataset_train = PNEDataset(args.train_data_dir, args.train_mask_dir,
                                          transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = PNEDataset(args.valid_data_dir, args.valid_mask_dir,
                                        transforms=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = PNEDataset(args.test_data_dir, args.test_mask_dir, transforms=None,
                                         normalization=args.normalization)
        criterion = torch_dice_coef_loss
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion)

if __name__ == '__main__':
    args = Args()
    main(args)