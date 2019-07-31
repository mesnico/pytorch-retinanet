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
from dataloader import collate_fn, AspectRatioBasedSampler, BalancedSampler, UnNormalizer, Normalizer
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
    parser.add_argument('--bs', help='Batch size', type=int, default=64)
    parser.add_argument('--net', help='Network to use', default='fasterrcnn')

    parser.add_argument('--log_interval', help='Iterations before outputting stats', type=int, default=1)
    parser.add_argument('--checkpoint_interval', help='Iterations before saving an intermediate checkpoint', type=int,
                        default=80)
    parser.add_argument('--iterations', type=int, help='Iterations for every batch', default=32)

    parser = parser.parse_args(args)

    # This becomes the minibatch size
    parser.bs = parser.bs // parser.iterations
    print('With {} iterations the effective batch size is {}'.format(parser.iterations, parser.bs))

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
                                       [ToTensor(), Augment(), Resizer(min_side=600, max_side=1000)]))
        dataset_val = OidDataset(parser.data_path, subset='validation',
                                 transform=Compose([ToTensor(), Resizer(min_side=600, max_side=1000)]))

    elif parser.dataset == 'dummy':
        # dummy dataset used only for debugging purposes
        dataset_train = DummyDataset(transform=Compose([ToTensor(), Resizer()]))
        # dataset_val = DummyDataset(transform=Compose([ToTensor(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = BalancedSampler(dataset_train, batch_size=parser.bs, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collate_fn, batch_sampler=sampler)

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
        log_losses_mean = {}
        running_loss_sum = 0

        data_progress = tqdm.tqdm(dataloader_train)
        old_tensors_set = {}
        optimizer.zero_grad()
        for minibatch_idx, data in enumerate(data_progress):

            images, targets = data

            images = list(image.cuda().float() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            #images, targets = images.cuda(), targets.cuda()

            loss_dict = model(images, targets)
            #classification_loss = classification_loss.mean()
            #regression_loss = regression_loss.mean()
            #loss = classification_loss + regression_loss
            loss = sum(loss for loss in loss_dict.values())
            monitor_loss = loss.clone().detach()
            loss /= parser.iterations
            running_loss_sum += float(loss.item())

            loss.backward()

            if len(log_losses_mean) == 0:
                log_losses_mean = clone_tensor_dict(loss_dict)
                log_losses_mean['total_loss'] = float(monitor_loss.item())
            else:
                loss_dict['total_loss'] = monitor_loss
                log_losses_mean = Counter(clone_tensor_dict(loss_dict)) + Counter(log_losses_mean)

            if (minibatch_idx + 1) % parser.iterations == 0:
                data_progress.set_postfix(dict(it=minibatch_idx // parser.iterations, loss=running_loss_sum))

                # all minibatches have been accumulated. Zero the grad
                optimizer.step()
                optimizer.zero_grad()
                running_loss_sum = 0

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # loss_hist.append(float(loss))
            # epoch_loss.append(float(loss))

            if (minibatch_idx + 1) % (parser.log_interval * parser.iterations) == 0:
                # compute the mean
                log_losses_mean = {k: (v / (parser.log_interval * parser.iterations)) for k, v in log_losses_mean.items()}

                logger.add_scalars("logs/losses", log_losses_mean,
                                   epoch_num * len(dataloader_train) + minibatch_idx)
                log_losses_mean = {}
            # print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            if (minibatch_idx + 1) % (parser.checkpoint_interval * parser.iterations) == 0:
                # Save an intermediate checkpoint
                save_checkpoint({
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_num
                }, experiment_fld, overwrite=True)

            if (minibatch_idx + 1) % (1 * parser.iterations) == 0:
                # flush cuda memory every tot iterations
                torch.cuda.empty_cache()

            '''for img in images:
                del img
            for tgt in targets:
                for t in tgt.values():
                    del t
            del loss'''

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
