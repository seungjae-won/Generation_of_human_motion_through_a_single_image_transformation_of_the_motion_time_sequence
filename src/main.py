from google.colab import drive
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import glob
import torchvision
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.functional as F
import torch.autograd as autograd
from model import *
from util import *
import argparse
from torch.autograd import Variable
from train import *


# Parser 생성하기
parser = argparse.ArgumentParser(description='Human motion classification', 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', default=2e-4, type=float, dest='lr')
parser.add_argument('--batch_size', default=4, type=int, dest='batch_size')
parser.add_argument('--num_epoch', default=400, type=int, dest='num_epoch')

parser.add_argument('--train_data_dir', default='/content/drive/My Drive/human_motion/traindata', type=str, dest='train_data_dir')
parser.add_argument('--test_data_dir', default='/content/drive/My Drive/human_motion/testdata', type=str, dest='test_data_dir')
parser.add_argument('--ACGAN_ckpt_dir', default='/content/drive/My Drive/human_motion/checkpoint', type=str, dest='ACGAN_ckpt_dir')
parser.add_argument('--ACGAN_log_dir', default='/content/drive/My Drive/human_motion/log', type=str, dest='ACGAN_log_dir')
parser.add_argument('--ACGAN_figure_dir', default='/content/drive/My Drive/human_motion/figure', type=str, dest='ACGAN_figure_dir')

# parser.add_argument('--oversampling_ckpt_dir', default='/content/drive/My Drive/human_motion/checkpoint', type=str, dest='oversampling_ckpt_dir')
# parser.add_argument('--oversampling_log_dir', default='/content/drive/My Drive/human_motion/log', type=str, dest='oversampling_log_dir')
# parser.add_argument('--oversampling_figure_dir', default='/content/drive/My Drive/human_motion/figure', type=str, dest='oversampling_figure_dir')
# parser.add_argument('--weight_balancing_ckpt_dir', default='/content/drive/My Drive/human_motion/checkpoint', type=str, dest='weight_balancing_ckpt_dir')
# parser.add_argument('--weight_balancing_log_dir', default='/content/drive/My Drive/human_motion/log', type=str, dest='weight_balancing_log_dir')
# parser.add_argument('--weight_balancing_figure_dir', default='/content/drive/My Drive/human_motion_figure', type=str, dest='weight_balancing_figure_dir')
# parser.add_argument('--feature_gan_ckpt_dir', default='/content/drive/My Drive/human_motion/checkpoint', type=str, dest='feature_gan_ckpt_dir')
# parser.add_argument('--feature_gan_log_dir', default='/content/drive/My Drive/human_motion/log', type=str, dest='feature_gan_log_dir')
# parser.add_argument('--lstm_retrain_ckpt_dir', default='/content/drive/My Drive/human_motion/checkpoint', type=str, dest='lstm_retrain_ckpt_dir')
# parser.add_argument('--lstm_retrain_log_dir', default='/content/drive/My Drive/human_motion/log', type=str, dest='lstm_retrain_log_dir')
# parser.add_argument('--lstm_retrain_figure_dir', default='/content/drive/My Drive/human_motion/figure', type=str, dest='lstm_retrain_figure_dir')


parser.add_argument('--sequence_length', default=100, type=int, dest='sequence_length')
parser.add_argument('--input_size', default=100, type=int, dest='input_size')
parser.add_argument('--num_classes', default=12, type=int, dest='num_classes')
parser.add_argument('--network', default='ACGAN', type=str, dest='network')
parser.add_argument('--mode', default='train', type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')


PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.gan_train()
    elif ARGS.mode == 'test':
        TRAINER.test()
    else:
        print('='*40)
        print('The entered "mode" does not exist')
        print('='*40)

if __name__ == '__main__':
    main()

