#!/usr/bin/env python

# [START all]
# [START libraries]
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../shared/')

import utils
import engine
import HSCGalaxyDataset
import argparse

from zoobot.pytorch.training import finetune

import json

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import tv_tensors
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2 as T
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
# [END libraries]

# [START initial settings]
if torch.cuda.is_available():
    TORCH_DEVICE = 'cuda'
    GPU_COUNT = torch.cuda.device_count()
    print('Device: {}, Number of GPUs: {}'.format(TORCH_DEVICE, GPU_COUNT))
else:
    TORCH_DEVICE = 'cpu'
# [END initial settings]

# [START classes and functions]
def get_transform(train):
    transforms = []

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))

    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))

    return T.Compose(transforms)


def get_dataloader_dict(train_df, val_df, image_dir, band, is_colour, batch_size):
    image_datasets = {}

    image_datasets['train'] = HSCGalaxyDataset.FRCNNGalaxyDataset(
        dataframe=train_df,
        image_dir=image_dir,
        band=band,
        colour=is_colour,
        transforms=get_transform(train=True)
    )
    
    image_datasets['val'] = HSCGalaxyDataset.FRCNNGalaxyDataset(
        dataframe=val_df,
        image_dir=image_dir,
        band=band,
        colour=is_colour,
        transforms=get_transform(train=False)
    )
    
    return {x: torch.utils.data.DataLoader(
        image_datasets[x], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=utils.collate_fn,
    ) for x in ['train', 'val']}


# Faster R-CNN requires a backbone with an attached FPN
class Resnet50WithFPN(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module,):
        super(Resnet50WithFPN, self).__init__()

        self.backbone = backbone

        # Extract 4 main layers (note: FRCNN needs this particular name mapping for return nodes)
        self.body = create_feature_extractor(
            self.backbone.encoder, 
            return_nodes={f'layer{k}': str(v) for v, k in enumerate([1, 2, 3, 4])}
        )
        
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 1, 300, 300)
        inp = inp.to(TORCH_DEVICE)
        
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def get_model(ckpt, num_classes=3, trainable_layers=0, image_mean=[0.485], image_std=[0.229]):
    """
    Creates the model object for Faster R-CNN

    Args:
      ckpt (str): path to checkpoint for the Zoobot backbone
      num_classes (int): number of classes the detector should output, 
        must include a class for the background
      trainable_layers (int): number of blocks of the classification backbone,
        counted from top, that should be made trainable
        e.g. 0 - all blocks fixed, 1 - 'backbone.body.conv1' trainable

    Returns:
      Faster R-CNN model

    """

    # load Zoobot backbone
    backbone = finetune.FinetuneableZoobotClassifier(checkpoint_loc=ckpt, num_classes=num_classes)

    # Build the model
    model = FasterRCNN(Resnet50WithFPN(backbone), num_classes=num_classes)

    # change the Transform layer to fit for single channel images
    grcnn = GeneralizedRCNNTransform(
        min_size=800, 
        max_size=1333, 
        image_mean=image_mean, 
        image_std=image_std
    )
    model.transform = grcnn

    # make sure, backbone layers are freezed after creating the model
    for name, parameter in model.named_parameters():
        if name.startswith('backbone.body.'):
            parameter.requires_grad = False
    
    # unfreeze selected layers
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    
    layers_to_train = [
        'backbone.backbone.encoder.layer4', 
        'backbone.backbone.encoder.layer3', 
        'backbone.backbone.encoder.layer2', 
        'backbone.backbone.encoder.layer1', 
        'backbone.backbone.encoder.conv1'
    ][:trainable_layers]
    
    if trainable_layers == 5:
        layers_to_train.append('backbone.backbone.encoder.bn1')
    for layer in layers_to_train:
        for name, parameter in model.named_parameters():
            if name.startswith(layer):
                parameter.requires_grad_(True)

    return model

# [END classes and functions]

# [START Main]
def main():
    # Parse arguments from cmd
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file', help='Config file as JSON', type=str)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # some checks, create directories if necessary
    # model logs
    if os.path.exists(config['log_dir']):
        for band in config['bands']:
            try:
                os.makedirs(config['log_dir'] + 'logs_eval/' + band)
            except FileExistsError:
                pass
            
            for k in range(config['k_folds']):
                try:
                    os.makedirs(config['log_dir'] + 'logs_eval/' + band + '/' + str(k))
                except FileExistsError:
                    pass
            
            try:
                os.makedirs(config['log_dir'] + 'logs_train/' + band)
            except FileExistsError:
                pass
            
            for k in range(config['k_folds']):
                try:
                    os.makedirs(config['log_dir'] + 'logs_train/' + band + '/' + str(k))
                except FileExistsError:
                    pass
        
        print('OK - Output directory for logs and checkpoints exists.')
        
    else:
        print('WARNING - Output directory for logs and checkpoints DOES NOT exist.')
        os.mkdir(config['log_dir'])
        os.mkdir(config['log_dir'] + 'logs_eval/')
        os.mkdir(config['log_dir'] + 'logs_train/')
        
        for band in config['bands']:
            try:
                os.makedirs(config['log_dir'] + 'logs_eval/' + band)
            except FileExistsError:
                pass
            
            for k in range(config['k_folds']):
                try:
                    os.makedirs(config['log_dir'] + 'logs_eval/' + band + '/' + str(k))
                except FileExistsError:
                    pass
            
            try:
                os.makedirs(config['log_dir'] + 'logs_train/' + band)
            except FileExistsError:
                pass
            
            for k in range(config['k_folds']):
                try:
                    os.makedirs(config['log_dir'] + 'logs_train/' + band + '/' + str(k))
                except FileExistsError:
                    pass
    
        print('OK - Output directory for logs and checkpoints created.')

    # Pre-trained model checkpoint
    if os.path.exists(config['pretrained_ckpt']):
        print('OK - Pre-trained model checkpoint exists.')
    else:
        print('ERROR - Pre-trained model checkpoint is MISSING.')
    
    # Table with training data
    if os.path.exists(config['data_table']):
        print('OK - Parquet-file with training data exists.')
    else:
        print('ERROR - Parquet-file with training data is MISSING.')
    
    # Directory with image data
    if os.path.exists(config['image_dir']):
        print('OK - Directory with image data exists.')
    else:
        print('ERROR - Directory with image data is MISSING.')

    # [START run prepartion]
    # Galaxy catalogue
    df = pd.read_parquet(config['data_table'], engine='pyarrow')
    # [END run prepartion]

    # [START Training]
    # torch.cuda.empty_cache() # empty GPU cache
    
    # Define the K-fold Cross Validator
    kf = KFold(n_splits=config['k_folds'], shuffle=True)
    X = df['local_ids'].unique()
    
    for band in config['bands']:
        
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            
            # initialise Tensorboard writer
            tb_log_dir = config['log_dir'] + 'logs_train/{}/{}/'.format(band, i)
            writer = SummaryWriter(log_dir=tb_log_dir)
            
            # get the model
            zoobot = get_model(
                ckpt=config['pretrained_ckpt'],
                num_classes=config['num_classes'],
                trainable_layers=config['trainable_layers'],
                image_mean=config['image_mean'][band],
                image_std=config['image_std'][band],
            )      
            # move model to the right device and using all available GPUs
            model = nn.DataParallel(zoobot)
            model.to(TORCH_DEVICE)
            
            # construct an optimizer
            params = [p for p in model.parameters() if p.requires_grad]
            # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.00005)
            
            # and a learning rate scheduler, comment for Adam
            # lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #     optimizer,
            #     step_size=3,
            #     gamma=0.1
            # )

            # load data
            dataloader_dict = get_dataloader_dict(
                train_df=df[(df['local_ids'].isin(X[train_index]))],
                val_df=df[(df['local_ids'].isin(X[test_index]))],
                image_dir=config['image_dir']+'{}-band/'.format(band),
                band=band,
                is_colour=False,
                batch_size=config['batch_size'],
            )
    
            # let's train
            for epoch in range(config['epochs']):
                # train for one epoch, printing every 10 iterations
                engine.train_one_epoch(
                    model, 
                    optimizer, 
                    dataloader_dict['train'], 
                    device=TORCH_DEVICE, 
                    epoch=epoch, 
                    print_freq=10,
                    scaler=None,
                    tb_writer=writer
                    # tb_writer=None
                )
                
                # update the learning rate
                # lr_scheduler.step()
            
                engine.evaluate_loss(
                    model, 
                    dataloader_dict['val'], 
                    device=TORCH_DEVICE, 
                    epoch=epoch, 
                    tb_writer=writer
                    # tb_writer=None
                )
            
                # evaluate on the test dataset
                engine.evaluate(
                    model, 
                    dataloader_dict['val'], 
                    device=TORCH_DEVICE,
                    epoch=epoch, 
                    tb_writer=writer
                    # tb_writer=None
                )
        
                # save model for each epoch
                model_save_path = config['log_dir'] + '{}_{}_FRCNN_fold_{}_epoch_{}.pth'.format(config['model_name'], band, i, epoch)
                torch.save(model.state_dict(), model_save_path)
    # [END Training]

# [END Main]

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit(1)

# [END all]