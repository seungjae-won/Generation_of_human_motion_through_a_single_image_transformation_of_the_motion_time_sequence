import os
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
from IPython.display import HTML
from matplotlib import rcParams
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def image_to_motion_sequence(image, sequence_length):
    size = sequence_length
    
    motion_sequences = []
    for m in range(len(image)):
        motion_sequence = []
        count = 0
        for i in range(20):
            if i != 0 and i != 1 and i != 12 and i != 16:
                motion = image[m][:,:,count:count+13]
                count+=13
            else:
                motion = image[m][:,:,count:count+12]
                count+=12

            motion = np.array(motion)
            motion_sequence.append(motion.mean(axis=-1))
        motion_sequences.append(motion_sequence)
    
    motion_sequences = np.array(motion_sequences)
    motion_sequences = np.transpose(motion_sequences,(0,2,3,1))
    
    return motion_sequences


def make_gif_file(motion_sequences, sequence_length, figure_dir):
    
    label_list = ['Lift outstretched arms', 'Duck', 'Push right', 
                  'Goggles', 'Wind it up', 'Shoot', 'Bow', 'Throw', 
                  'Had enough', 'Change weapon', 'Beat both', 'Kick']
    
    for i in range(len(motion_sequences)):
        class_num = i
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_axis_off()
        
        connect_list = [[0,1],[1,2],[2,3],[2,4],[2,8],[0,12],[0,16],[4,5],[5,6],
                        [6,7],[8,9],[9,10],[10,11],[12,13],[13,14],[14,15],[16,17],[17,18],[18,19]]
        
        for connect in connect_list:
            ax.plot([motion_sequences[class_num][0][0][connect[0]],motion_sequences[class_num][0][0][connect[1]]], 
                    [motion_sequences[class_num][0][2][connect[0]],motion_sequences[class_num][0][2][connect[1]]], 
                    [motion_sequences[class_num][0][1][connect[0]],motion_sequences[class_num][0][1][connect[1]]], 
                    label='parametr0c curve', marker='o')
            
        ax.set_title(label_list[class_num]+' {}/{}'.format(0,sequence_length))

        def update_graph(num):
            plt.cla()
            plt.clf()
            ax = fig.gca(projection='3d')
            ax.set_axis_off()
            for connect in connect_list:
                ax.plot([motion_sequences[class_num][num][0][connect[0]],motion_sequences[class_num][num][0][connect[1]]], 
                        [motion_sequences[class_num][num][2][connect[0]],motion_sequences[class_num][num][2][connect[1]]], 
                        [motion_sequences[class_num][num][1][connect[0]],motion_sequences[class_num][num][1][connect[1]]], 
                        label='parametr0c curve', marker='o')
            ax.set_title(label_list[class_num]+' {}/{}'.format(num,sequence_length))
            
            ax.view_init(elev=10., azim=270)

        ani = matplotlib.animation.FuncAnimation(fig, update_graph, sequence_length, interval=400, blit=False)
        save_path = os.path.join(figure_dir,'fake_motion_'+label_list[class_num]+'.gif')
        ani.save(save_path, writer='imagemagick', fps=10)





def save_model(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    torch.save({'net':net.state_dict(), 'optim': optim.state_dict()},
                '%s/model_epoch_%d.pth'%(ckpt_dir,epoch))



def load_model(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
        
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()
    print('%s/%s' %(ckpt_dir, ckpt_lst[-1]))
    dict_model = torch.load('%s/%s' %(ckpt_dir, ckpt_lst[-1]))
    
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    return net, optim

# source
# https://github.com/hanyoseob/pytorch-StarGAN/blob/master/utils.py
class Parser:
    def __init__(self, parser):
        self.__parser = parser
        self.__args = parser.parse_args()

    def get_parser(self):
        return self.__parser

    def get_arguments(self):
        return self.__args

    def print_args(self, name='PARAMETER TABLES'):
        params_dict = vars(self.__args)
        print('\n\n')
        print('----' * 10)
        print('{0:^40}'.format(name))
        print('----' * 10)
        for k, v in sorted(params_dict.items()):
            if '__' not in str(k):
                print('{}'.format(str(k)) + ' : ' + ('{0:>%d}' % (35 - len(str(k)))).format(str(v)))
        print('----' * 10)
        print('\n\n')

