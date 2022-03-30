'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''


"""Single-stream Learner"""
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import ResNet as models
import numpy as np
from torch.utils import model_zoo
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=8e-4, help='learning rate (default: 8e-4')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--method', type=str, default='single_stream')
parser.add_argument('--opt', type=str, default='no_ent')
parser.add_argument('--backbone', type=str, default='resnet101')
parser.add_argument('--dataset', type=str, default='visda-c')
parser.add_argument('--data_root', type=str, default='/')
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--rand_id', type=int, default=0)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--th', type=float, default=0.95)
parser.add_argument('--div_weight', type=float, default=0.4)
parser.add_argument('--st', type=int, default=0)
parser.add_argument('--runs', type=int, default=0)
args = parser.parse_args()

"""params"""

batch_size = args.batch
epochs = 1
lr = args.lr
momentum = 0.9
seed = 999
l2_decay = 5e-4
kwargs = {'num_workers': 0, 'pin_memory': False}
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True

"""loaders"""

if args.dataset == 'visda-c':
    source_name = "train"
    target_name = "validation"
    num_classes = 12
    source_file = './data/visda-c/train_list.txt'
    target_test_file = './data/visda-c/validation_list.txt'

elif args.dataset == 'fashion':
    num_classes = 6
    source_name = 'fashion_mnist'
    target_name = 'deepfashion'
    source_file = './data/fashion/fashion_mnist_train_list.txt'
    target_test_file = './data/fashion/deepfashion_train_list.txt'

source_loader = data_loader.load_training_from_list(args.data_root, source_file, batch_size, kwargs, shuffle=True)
target_test_loader = data_loader.load_testing_from_list(args.data_root, target_test_file, batch_size, kwargs, shuffle=False)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
"""target plot"""
file_path = './data/{}/{}_{}.txt'.format(args.dataset, target_name, args.rand_id)
target_train_loader = data_loader.load_training_strong_weak(args.data_root, file_path, batch_size, kwargs, shuffle=False,
                                                            return_test_img=True)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_test_loader.dataset)
len_source_loader = len(source_loader)
len_target_loader = len(target_train_loader)


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def load_pretrain(net):
    if '18' in args.backbone:
        url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    if '50' in args.backbone:
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    elif '101' in args.backbone:
        url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = net.state_dict()
    for k, v in model_dict.items():
        if not "cls_fc" in k and not "num_batches_tracked" in k and not "prototype" in k:
            model_dict[k] = pretrained_dict[k[k.find(".") + 1:]]
    net.load_state_dict(model_dict)
    return net


def get_optim(net, LEARNING_RATE, parallel=False, optim='adam'):
    net = net.module if parallel else net
    if optim == 'adam':
        optimizer = torch.optim.Adam([
            {'params': net.sharedNet.parameters(), 'lr': LEARNING_RATE / 100},
            {'params': net.prototype.parameters(), 'lr': LEARNING_RATE},
            {'params': net.prototype_bn.parameters(), 'lr': LEARNING_RATE},
            {'params': net.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, weight_decay=l2_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': net.sharedNet.parameters(), 'lr': LEARNING_RATE / 10},
            {'params': net.prototype.parameters(), 'lr': LEARNING_RATE},
            {'params': net.prototype_bn.parameters(), 'lr': LEARNING_RATE},
            {'params': net.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, weight_decay=l2_decay)
    return optimizer


def train(model):
    optimizer = get_optim(model, LEARNING_RATE=lr, parallel=True, optim=args.optim)

    data_source_iter = iter(source_loader)
    data_target_iter = iter(target_train_loader)

    i = 1

    batch_acc = []

    while i <= len_target_loader:
        model.train()

        try:
            source_data, source_label = data_source_iter.next()
        except:
            data_source_iter = iter(source_loader)
            source_data, source_label = data_source_iter.next()

        """source label loss"""
        clabel_src, _ = model(source_data.cuda())
        label_loss = F.nll_loss(clabel_src.log(), source_label.cuda())

        del clabel_src

        (target_data_w, target_data_s, target_data_test), target_label = data_target_iter.next()
        if i % len_target_loader == 0:
            data_target_iter = iter(target_train_loader)

        _, clabel_tgt_w = model(target_data_w.cuda())

        pseudo_label = torch.softmax(clabel_tgt_w.detach().cpu(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.th).float()

        if args.opt.upper() == 'INFOMAX':
            # w/o CroDoBo
            clabel_tgt_w = F.softmax(clabel_tgt_w, dim=1)
            _, clabel_tgt_s = model(target_data_s.cuda())
            pb_pred_tgt = clabel_tgt_w.sum(dim=0)
            pb_pred_tgt = 1.0 / pb_pred_tgt.sum() * pb_pred_tgt
            target_div_loss = - torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))
            target_entropy_loss = -torch.mean((clabel_tgt_w * torch.log(clabel_tgt_w + 1e-6)).sum(dim=1))
            del clabel_tgt_w
            del pb_pred_tgt
            Lu = (F.cross_entropy(clabel_tgt_s, targets_u.cuda(), reduction='none') * mask.cuda()).mean()
            total_loss = label_loss + 1.0 * target_entropy_loss - args.div_weight * target_div_loss + Lu

        elif args.opt.upper() == 'INFOMAX_S':
            # w/o CroDoBo with RandAug on l_div, l_ent
            _, clabel_tgt_s = model(target_data_s.cuda())
            Lu = (F.cross_entropy(clabel_tgt_s, targets_u.cuda(), reduction='none') * mask.cuda()).mean()
            clabel_tgt_s = F.softmax(clabel_tgt_s, dim=1)
            pb_pred_tgt = clabel_tgt_s.sum(dim=0)
            pb_pred_tgt = 1.0 / pb_pred_tgt.sum() * pb_pred_tgt
            target_div_loss = - torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))
            target_entropy_loss = -torch.mean((clabel_tgt_s * torch.log(clabel_tgt_s + 1e-6)).sum(dim=1))
            total_loss = label_loss + 1.0 * target_entropy_loss - args.div_weight * target_div_loss + Lu

        elif 'NO_ENT' in args.opt.upper():
            # w/o CroDoBo, remove l_ent
            _, clabel_tgt_s = model(target_data_s.cuda())
            Lu = (F.cross_entropy(clabel_tgt_s, targets_u.cuda(), reduction='none') * mask.cuda()).mean()
            pred = F.softmax(clabel_tgt_w, dim=1)
            pb_pred_tgt = pred.sum(dim=0)
            pb_pred_tgt = 1.0 / pb_pred_tgt.sum() * pb_pred_tgt
            target_div_loss = - torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))
            total_loss = label_loss - args.div_weight * target_div_loss + Lu

        elif 'NO_DIV' in args.opt.upper():
            # w/o CroDoBo, remove l_div
            _, clabel_tgt_s = model(target_data_s.cuda())
            Lu = (F.cross_entropy(clabel_tgt_s, targets_u.cuda(), reduction='none') * mask.cuda()).mean()
            pred = F.softmax(clabel_tgt_w, dim=1)
            target_entropy_loss = -torch.mean((pred * torch.log(pred + 1e-6)).sum(dim=1))
            total_loss = label_loss + 1.0 * target_entropy_loss + Lu

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        """Current Query Eval"""
        acc_tgt = test_batch(model, target_data_test, target_label)
        batch_acc.append(acc_tgt.item())
        print('Query %d Acc %.2f' % (i, acc_tgt))

        i = i + 1

    return batch_acc


def test_batch(net, batch_data, batch_label):
    net.eval()
    with torch.no_grad():
        batch_out, _ = net(batch_data.cuda())
        batch_out = batch_out.data.max(1)[1]
        batch_acc_avg = batch_out.eq(batch_label.cuda()).cpu().sum() * 100. / len(batch_label)
    return batch_acc_avg


def test_visda(net):
    """For VisDA-C sample acc: correct/all; category acc: mean(class_acc)"""
    net.eval()
    correct = 0
    total = 0

    Dict_all = np.zeros(num_classes)
    Dict_acc = np.zeros(num_classes)

    with torch.no_grad():
        for batch_data, batch_label in target_test_loader:
            batch_out, _ = net(batch_data.cuda())
            pred = batch_out.data.cpu().max(1)[1]
            total += batch_label.size(0)

            for j in range(batch_label.numpy().shape[0]):
                Dict_all[batch_label[j].item()] += 1

                if pred[j] == batch_label[j]:
                    Dict_acc[batch_label[j].item()] += 1
                    correct += 1

    for j in range(len(Dict_all)):
        Dict_acc[j] = Dict_acc[j] / Dict_all[j] * 100.

    sample_acc_all = correct * 100. / total

    return sample_acc_all, Dict_acc, Dict_all


def batch_figure(batch_data):
    fig, ax = plt.subplots(figsize=(5, 4))
    t = np.arange(len(batch_data))
    ax.set_xlabel('Batch Index')
    ax.set_ylim([0, 101])
    ax.plot(t, batch_data, color='royalblue', linewidth=1.5)
    return fig


def test(net):
    net.eval()
    total = 0
    correct_all = 0

    with torch.no_grad():
        for data, label in target_test_loader:
            label = label.long()
            batch_out, _ = net(data.cuda())
            pred = batch_out.data.cpu().max(1)[1]
            correct_all += pred.eq(label).cpu().sum().item()
            total += label.size(0)

    acc_all = 100. * correct_all / total
    return acc_all


if __name__ == '__main__':

    log_root = './log/%s/%s/%s/%.2f/div_%.2f' % (
        args.dataset, args.method, args.opt, args.th, args.div_weight)

    test_log_folder = os.path.join(log_root, './log_test_%s_%s_lr_%.5f_st_%d_rand_id_%d_batch_%d_run%d' % (
        args.backbone,
        args.optim, lr,
        args.st,
        args.rand_id, args.batch,
        args.runs))
    if not os.path.exists(test_log_folder):
        os.makedirs(test_log_folder)

    test_log = os.path.join(log_root, 'log_test_%s_%s_lr_%.5f_st_%d_rand_id_%d_batch_%d_run%d.txt' % (
        args.backbone,
        args.optim, lr,
        args.st,
        args.rand_id, args.batch,
        args.runs))

    model = models.MEDM_prototype(num_classes=num_classes, backbone=args.backbone)

    if 'densenet' not in args.backbone:
        model = load_pretrain(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    for epoch in range(1, epochs + 1):
        epoch_batch_acc = train(model)

        epoch_batch_log = np.array(epoch_batch_acc)
        np.save(os.path.join(test_log_folder, 'batch_{}_log_epoch_{}.npy'.format(args.batch, epoch)), epoch_batch_log)

        test_msg = ''
        test_msg += 'Online Acc: %.2f\n' % (np.array(epoch_batch_acc).mean())

        if args.dataset == 'visda-c':
            true_acc, class_acc, class_samples = test_visda(model)

            Dict_name = {0: 'plane', 1: 'bike', 2: 'bus', 3: 'car', 4: 'horse', 5: 'knife', 6: 'motor',
                         7: 'person', 8: 'plant', 9: 'sktboard', 10: 'train', 11: 'truck'}

            test_msg += 'One-pass Acc: sample-wise %.2f category-mean %.2f\n' % (true_acc, class_acc.mean())
            for j in range(12):
                test_msg += '%s %.2f ' % (Dict_name[j], class_acc[j])
            write_log(test_msg, test_log)
            print(test_msg)

        else:
            acc_all = test(model)

            test_msg = 'One-pass Acc: %.2f' % acc_all
