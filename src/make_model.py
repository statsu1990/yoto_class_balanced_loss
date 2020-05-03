import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import albumentations as alb
from albumentations.augmentations import transforms as albtr
from albumentations.pytorch import ToTensor as albToTensor

import torch.nn as nn
import torch.optim as optim

from data import cifar, torch_data_utils
from model import resnet, resnet_yoto, cb_loss
from train import training


def get_checkpoint(path):
    cp = torch.load(path, map_location=lambda storage, loc: storage)
    return cp

class Resnet_Imb_CE:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = nn.CrossEntropyLoss()
        vl_criterion = nn.CrossEntropyLoss()

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta0_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.0)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.0)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta09_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.9)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.9)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta099_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.99)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.99)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta0999_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta09999_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.9999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.9999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta099999_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=10)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.99999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.99999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_YOTO_ep100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = "Resnet_Imb_YOTO_ep100_v2_checkpoint" #None

    def get_model(self):
        param_ranges=((0.9, 0.99999),)
        params=(0.999,)
        param_dist='log1m_uniform'
        param_sampler = resnet_yoto.ParamSampler(param_ranges, params, param_dist)

        model = resnet_yoto.ResNet18(10, param_sampler)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=True)
        training.test_yoto(model, ts_loader, 
                  param_grids=((0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 
                                0.999995, 0.999999, 0.9999995, 0.9999999, 0.99999995, 0.99999999),),
                  filename_head=self.filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_YOTO_ep100_SameParamInBatch:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        param_ranges=((0.9, 0.99999),)
        params=(0.999,)
        param_dist='log1m_uniform'
        same_param_in_batch = True
        param_sampler = resnet_yoto.ParamSampler(param_ranges, params, param_dist, same_param_in_batch)

        model = resnet_yoto.ResNet18(10, param_sampler)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=True)
        training.test_yoto(model, ts_loader, 
                  param_grids=((0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 
                                0.999995, 0.999999, 0.9999995, 0.9999999, 0.99999995, 0.99999999),),
                  filename_head=self.filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_YOTO_ep100_WoFilm:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        param_ranges=((0.9, 0.99999),)
        params=(0.999,)
        param_dist='log1m_uniform'
        use_film = False
        param_sampler = resnet_yoto.ParamSampler(param_ranges, params, param_dist)

        model = resnet_yoto.ResNet18(10, param_sampler, use_film)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar10(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar10(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 10, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 10, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=True)
        #training.test_yoto(model, ts_loader, 
        #          param_grids=((0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 
        #                        0.999995, 0.999999, 0.9999995, 0.9999999, 0.99999995, 0.99999999),),
        #          filename_head=self.filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return


# to use cifar 100, have to change major_labels and minor_labels of training.ScoreCalculator.
class Resnet_Imb_CB_beta0_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.0)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.0)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta09_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.9)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.9)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta099_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.99)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.99)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta0999_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta09999_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.9999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.9999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta099999_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.99999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.99999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_YOTO_ep100_cifar100:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        param_ranges=((0.9, 0.99999),)
        params=(0.999,)
        param_dist='log1m_uniform'
        param_sampler = resnet_yoto.ParamSampler(param_ranges, params, param_dist)

        model = resnet_yoto.ResNet18(100, param_sampler)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.10,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=True)
        training.test_yoto(model, ts_loader, 
                  param_grids=((0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 
                                0.999995, 0.999999, 0.9999995, 0.9999999, 0.99999995, 0.99999999),),
                  filename_head=self.filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return


class Resnet_Imb_CB_beta0_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.0)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.0)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta09_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.9)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.9)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta099_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.99)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.99)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta0999_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta09999_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.9999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.9999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_CB_beta099999_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        model = resnet.ResNet18(num_classes=100)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.99999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.99999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=False)
        #training.test_yoto(model, ts_loader, 
        #          param_ranges=param_ranges, n_grid=11,
        #          filename_head=filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return

class Resnet_Imb_YOTO_ep100_cifar100_2:
    def __init__(self):
        self.set_config()

    def set_config(self):
        self.filename_head = self.__class__.__name__ + '_'
        self.checkpoint_path = None

    def get_model(self):
        param_ranges=((0.9, 0.99999),)
        params=(0.999,)
        param_dist='log1m_uniform'
        param_sampler = resnet_yoto.ParamSampler(param_ranges, params, param_dist)

        model = resnet_yoto.ResNet18(100, param_sampler)
        return model

    def get_dataset(self, return_target=True):
        DOWNLOAD = False

        # transformer
        tr_transformer = alb.Compose([
                                albtr.Flip(p=0.5),
                                albtr.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=15, p=0.5),
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        ts_transformer = alb.Compose([
                                albtr.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                albToTensor()
                                ])
        # dataset
        #usage_rate = (1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05,)
        usage_rate = (1,)*50 + (0.05,)*50
        seed = 2020
        tr_ds, tr_tg = cifar.get_dataset_cifar100(True, DOWNLOAD, torch_data_utils.ImgDataset, tr_transformer, usage_rate, seed, return_target)
        ts_ds, ts_tg = cifar.get_dataset_cifar100(False, DOWNLOAD, torch_data_utils.ImgDataset, ts_transformer, None, None, return_target)

        if return_target:
            return tr_ds, ts_ds, tr_tg, ts_tg
        else:
            return tr_ds, ts_ds

    def train_model(self, use_checkpoint=False, fine_turning=False):
        # data
        tr_ds, ts_ds, tr_tg, ts_tg = self.get_dataset(return_target=True)

        # checkpoint
        if use_checkpoint:
            CP = get_checkpoint(self.checkpoint_path)
        else:
            CP = None

        # model
        model = self.get_model()
        if CP is not None:
            model.load_state_dict(CP['state_dict'])

        ## training
        TR_BATCH_SIZE = 128
        TS_BATCH_SIZE = 512
        tr_loader = torch_data_utils.get_dataloader(tr_ds, TR_BATCH_SIZE)
        ts_loader = torch_data_utils.get_dataloader(ts_ds, TS_BATCH_SIZE, shuffle=False)

        LR = 0.1
        opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        if CP is not None:
            if not fine_turning:
                opt.load_state_dict(CP['optimizer'])
        tr_criterion = cb_loss.ClassBalanced_CELoss(tr_tg, 100, beta=0.999)
        vl_criterion = cb_loss.ClassBalanced_CELoss(ts_tg, 100, beta=0.999)

        grad_accum_steps = 1
        start_epoch = 0 if CP is None or fine_turning else CP['epoch']
        EPOCHS = 100
        warmup_epoch=0
        step_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[51, 86, 101], gamma=0.1) #learning rate decay

        model = training.train_model(model, tr_loader, ts_loader, 
                                     opt, tr_criterion, vl_criterion, 
                                     grad_accum_steps, start_epoch, EPOCHS, 
                                     warmup_epoch, step_scheduler, self.filename_head, use_yoto=True)
        training.test_yoto(model, ts_loader, 
                  param_grids=((0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 
                                0.999995, 0.999999, 0.9999995, 0.9999999, 0.99999995, 0.99999999),),
                  filename_head=self.filename_head)

        # save
        #torch.save(model.state_dict(), filename_head + '_model')

        return
