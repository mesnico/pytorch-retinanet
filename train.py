import time
import os
import copy
import argparse
import pdb
import collections
import sys
import tqdm

import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from tensorboardX import SummaryWriter

from datasets import OidDataset, DummyDataset    # TODO: only openimages is supported at the moment
from dataloader import collate_fn, AspectRatioBasedSampler, BalancedAspectRatioBasedSampler, UnNormalizer, Normalizer
from transforms import Compose, RandomHorizontalFlip, Resizer, ToTensor, Augment
from create_model import create_model
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval

import gc

# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def clone_tensor_dict(d):
    return {k: float(v.item()) for k, v in d.items()}


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv, coco or openimages')
    parser.add_argument('--data_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--resume', help='Checkpoint to load')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--net', help='Network to use', default='fasterrcnn')

    parser.add_argument('--log_interval', help='Iterations before outputting stats', type=int, default=50)
    parser.add_argument('--checkpoint_interval', help='Iterations before saving an intermediate checkpoint', type=int,
                        default=5000)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':
        raise NotImplementedError()
        if parser.data_path is None:
            raise ValueError('Must provide --data_path when training on COCO,')

        dataset_train = CocoDataset(parser.data_path, set_name='train2017',
                                    transform=Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.data_path, set_name='val2017',
                                  transform=Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        raise NotImplementedError()

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'openimages':
        if parser.data_path is None:
            raise ValueError('Must provide --data_path when training on OpenImages')

        dataset_train = OidDataset(parser.data_path, subset='train',
                                   transform=Compose(
                                       [ToTensor(), Augment(), Resizer()]))
        dataset_val = OidDataset(parser.data_path, subset='validation',
                                 transform=Compose([ToTensor(), Resizer()]))

    elif parser.dataset == 'dummy':
        # dummy dataset used only for debugging purposes
        dataset_train = DummyDataset(transform=Compose([ToTensor(), Resizer()]))
        # dataset_val = DummyDataset(transform=Compose([ToTensor(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = BalancedAspectRatioBasedSampler(dataset_train, batch_size=parser.bs, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=12, collate_fn=collate_fn, batch_sampler=sampler)

    # if dataset_val is not None:
    #    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #    dataloader_val = DataLoader(dataset_val, num_workers=12, collate_fn=collate_fn, batch_sampler=sampler_val)

    # Create the model
    model = create_model(dataset_train.num_classes(), parser)

    # Create the experiment folder
    experiment_fld = 'experiment_{}_{}_resnet{}_{}'.format(parser.net, parser.dataset, parser.depth,
                                                        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    experiment_fld = os.path.join('outputs', experiment_fld)
    if not os.path.exists(experiment_fld):
        os.makedirs(experiment_fld)

    logger = SummaryWriter(experiment_fld)

    use_gpu = True

    if use_gpu:
        model = model.cuda()

    model = torch.nn.DataParallel(model).cuda()
    model.training = True

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Load checkpoint if needed
    start_epoch = 0
    if parser.resume:
        print('Loading checkpoint {}'.format(parser.resume))
        checkpoint = torch.load(parser.resume)
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Checkpoint loaded!')

    loss_hist = collections.deque(maxlen=500)

    model.train()
    # model.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in tqdm.trange(start_epoch, parser.epochs):

        model.train()
        # model.module.freeze_bn()

        epoch_loss = []
        losses_mean = {}
        '''loss_mean = 0
        regression_loss_mean = 0
        classification_loss_mean = 0'''

        data_progress = tqdm.tqdm(dataloader_train)
        old_tensors_set = {}
        for iter_num, data in enumerate(data_progress):
            if (iter_num + 1) % 5000000 == 0:
                n_tensors = 0
                tensors_set = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)) and obj.is_cuda:
                            #print(type(obj), obj.size())
                            n_tensors+=1
                            #tensors_set.append(obj)
                    except:
                        pass
                #tensors_set = set(tensors_set)
                #diff = tensors_set.difference(old_tensors_set)
                #for el in diff:
                #    print(type(el), el.size())
                #old_tensors_set = tensors_set
                print('Number of active tensors: {}; increment: {}'.format(n_tensors, None))

            optimizer.zero_grad()

            images, targets = data

            images = list(image.cuda().float() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            #images, targets = images.cuda(), targets.cuda()

            loss_dict = model(images, targets)
            #classification_loss = classification_loss.mean()
            #regression_loss = regression_loss.mean()
            #loss = classification_loss + regression_loss
            loss = sum(loss for loss in loss_dict.values())
            if len(losses_mean) == 0:
                losses_mean = clone_tensor_dict(loss_dict)
                losses_mean['total_loss'] = float(loss.item())
            else:
                loss_dict['total_loss'] = loss
                losses_mean = Counter(clone_tensor_dict(loss_dict)) + Counter(losses_mean)

            # if bool(loss == 0):
            #	continue

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # loss_hist.append(float(loss))
            # epoch_loss.append(float(loss))

            data_progress.set_postfix(dict(loss=float(loss.item())))

            if (iter_num + 1) % parser.log_interval == 0:
                # compute the mean
                losses_mean = {k: (v / parser.log_interval) for k, v in losses_mean.items()}

                logger.add_scalars("logs/losses", losses_mean,
                                   epoch_num * len(dataloader_train) + iter_num)
                losses_mean = {}
            # print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            if (iter_num + 1) % parser.checkpoint_interval == 0:
                # Save an intermediate checkpoint
                save_checkpoint({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_num
                }, experiment_fld, overwrite=True)

            del loss

            if (iter_num + 1) % 50 == 0:
                # flush cuda memory every tot iterations
                torch.cuda.empty_cache()

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, model)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, model)

        # TODO: write evaluation code for openimages
        scheduler.step(np.mean(epoch_loss))

        save_checkpoint({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch_num,
        }, experiment_fld, overwrite=False)

    model.eval()


# torch.save(retinanet, 'model_final.pt'.format(epoch_num))


def save_checkpoint(data, path, overwrite=False):
    epoch = data['epoch']
    if overwrite:
        outfile = 'checkpoint.pth'
    else:
        outfile = 'checkpoint_epoch_{}.pth'.format(epoch)
    outfile = os.path.join(path, outfile)
    torch.save(data, outfile)


if __name__ == '__main__':
    main()
