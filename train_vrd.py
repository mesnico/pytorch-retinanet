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

from datasets import OidDatasetVRD, DummyDataset
from dataloader import collate_fn, AspectRatioBasedSampler, BalancedSampler, UnNormalizer, Normalizer
from transforms import Compose, RandomHorizontalFlip, Resizer, ToTensor, Augment
from models.create_model import create_detection_model
from models.vrd import VRD
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
    parser.add_argument('--resume', help='Checkpoint to load the model from')
    parser.add_argument('--resume_attr', help='Checkpoint to load the attributes model from')
    parser.add_argument('--resume_rel', help='Checkpoint to load the relationships from')
    parser.add_argument('--detector_snapshot', help='Detector snapshot')
    parser.add_argument('--finetune_detector', action='store_true', default=False, help='Enable finetuning the detector')
    parser.add_argument('--lr_step_size', type=int, default=20, help="After how many epochs the lr is decreased")
    parser.add_argument('--lr', type=int, default=1e-4, help="Initial learning rate")

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--bs', help='Batch size', type=int, default=64)
    parser.add_argument('--net', help='Network to use', default='fasterrcnn')
    parser.add_argument('--train_rel', action='store_true', default=False, help='Enable training relationships')
    parser.add_argument('--train_attr', action='store_true', default=False, help='Enable training attributes')

    parser.add_argument('--log_interval', help='Iterations before outputting stats', type=int, default=1)
    parser.add_argument('--checkpoint_interval', help='Iterations before saving an intermediate checkpoint', type=int,
                        default=80)
    parser.add_argument('--iterations', type=int, help='Iterations for every batch', default=32)

    parser = parser.parse_args()

    # asserts
    assert parser.train_rel or parser.train_attr, "You have to train one of attribute or relation networks!"
    assert not (not parser.train_rel and parser.resume_rel), "It is useless to load relationships when you do not train them!"
    assert not (not parser.train_attr and parser.resume_attr), "It is useless to load attributes when you do not train them!"

    # This becomes the minibatch size
    parser.bs = parser.bs // parser.iterations
    print('With {} iterations the effective batch size is {}'.format(parser.iterations, parser.bs))

    # Create the data loaders
    if parser.dataset == 'openimages':
        if parser.data_path is None:
            raise ValueError('Must provide --data_path when training on OpenImages')

        dataset_train = OidDatasetVRD(parser.data_path, subset='train',
                                   transform=Compose(
                                       [ToTensor(), Augment(), Resizer(min_side=600, max_side=1000)]))
        # dataset_val = OidDatasetVRD(parser.data_path, subset='validation',
        #                         transform=Compose([ToTensor(), Resizer(min_side=600, max_side=1000)]))

    elif parser.dataset == 'dummy':
        # dummy dataset used only for debugging purposes
        raise NotImplementedError()

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    # if training one of relationships or attributes, balance!
    # if not (parser.train_attr and parser.train_rel):
    print('Dataloader is using the BalancedSampler!')
    sampler_train = BalancedSampler(dataset_train, batch_size=parser.bs, train_rel=parser.train_rel, train_attr=parser.train_attr)
    dataloader_train = DataLoader(dataset_train, num_workers=8, collate_fn=collate_fn, batch_sampler=sampler_train)
    # dataloader_train = DataLoader(dataset_train, num_workers=8, batch_size=parser.bs, collate_fn=collate_fn, shuffle=True)

    # if dataset_val is not None:
    #    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    #    dataloader_val = DataLoader(dataset_val, num_workers=12, collate_fn=collate_fn, batch_sampler=sampler_val)

    # Create the detection model
    detector = create_detection_model(dataset_train.num_classes(), parser)

    # Create the experiment folder
    if parser.train_attr and parser.train_rel:
        mode = 'attr-and-rel'
    elif parser.train_attr:
        mode = 'only-attr'
    elif parser.train_rel:
        mode = 'only-rel'
    experiment_fld = 'vrd_{}_experiment_{}_{}_resnet{}_{}'.format(mode, parser.net, parser.dataset, parser.depth,
                                                        time.strftime("%Y%m%d%H%M%S", time.localtime()))
    experiment_fld = os.path.join('outputs', experiment_fld)
    if not os.path.exists(experiment_fld):
        os.makedirs(experiment_fld)

    logger = SummaryWriter(experiment_fld)

    use_gpu = True

    #if use_gpu:
    #    detector = detector.cuda()
    #    detector = torch.nn.DataParallel(detector).cuda()

    if parser.detector_snapshot:
        checkpoint = torch.load(parser.detector_snapshot)
        weights = checkpoint['model']
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        detector.load_state_dict(weights)
        print('Correctly loaded the detector checkpoint {}'.format(parser.detector_snapshot))

    # Create the VRD model given the detector
    model = VRD(detector, dataset=dataset_train, train_relationships=parser.train_rel,
                train_attributes=parser.train_attr, finetune_detector=parser.finetune_detector)
    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=parser.lr_step_size)

    # Load checkpoint if needed
    start_epoch = 0
    # load relationships
    if parser.resume_rel:
        print('Loading relationship checkpoint {}'.format(parser.resume_rel))
        rel_checkpoint = torch.load(parser.resume_rel)
        model.module.relationships_net.load_state_dict(rel_checkpoint['model_rel'])
        if not parser.resume_attr:
            print('Resuming also scheduler and optimizer...')
            start_epoch = rel_checkpoint['epoch']
            optimizer.load_state_dict(rel_checkpoint['optimizer'])
            scheduler.load_state_dict(rel_checkpoint['scheduler'])
    if parser.resume_attr:
        print('Loading attributes checkpoint {}'.format(parser.resume_attr))
        attr_checkpoint = torch.load(parser.resume_attr)
        model.module.attributes_net.load_state_dict(attr_checkpoint['model_attr'])
        if not parser.resume_rel:
            print('Resuming also scheduler and optimizer...')
            start_epoch = attr_checkpoint['epoch']
            optimizer.load_state_dict(attr_checkpoint['optimizer'])
            scheduler.load_state_dict(attr_checkpoint['scheduler'])
    if parser.resume:
        print('Loading both attributes and relationships models {}'.format(parser.resume))
        checkpoint = torch.load(parser.resume)
        model.module.relationships_net.load_state_dict(checkpoint['model_rel'])
        model.module.attributes_net.load_state_dict(checkpoint['model_attr'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    print('Checkpoint loaded!')

    loss_hist = collections.deque(maxlen=500)

    model.train()
    # model.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in tqdm.trange(start_epoch, parser.epochs):
        logger.add_scalar("learning_rate", optimizer.param_groups[0]['lr'],
                          epoch_num * len(dataloader_train))

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
                    'model_rel': model.module.relationships_net.state_dict() if parser.train_rel else None,
                    'model_attr': model.module.attributes_net.state_dict() if parser.train_attr else None,
                    'model_det': model.module.detector.state_dict() if parser.finetune_detector else None,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_num
                }, experiment_fld, overwrite=True)

            if (minibatch_idx + 1) % 5 == 0:
                # flush cuda memory every tot iterations
                torch.cuda.empty_cache()

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, model)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, model)

        # TODO: write evaluation code for openimages
        scheduler.step()

        save_checkpoint({
            'model_rel': model.module.relationships_net.state_dict() if parser.train_rel else None,
            'model_attr': model.module.attributes_net.state_dict() if parser.train_attr else None,
            'model_det': model.module.detector.state_dict() if parser.finetune_detector else None,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch_num
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
