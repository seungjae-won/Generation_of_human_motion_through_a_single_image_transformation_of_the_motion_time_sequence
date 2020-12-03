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
from dataset import *
from util import *
import argparse
from torch.autograd import Variable
from train import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class Train:
    
    def __init__(self, args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        
        self.train_data_dir = args.train_data_dir
        self.ACGAN_ckpt_dir = args.ACGAN_ckpt_dir
        self.ACGAN_log_dir = args.ACGAN_log_dir
        self.ACGAN_figure_dir = args.ACGAN_figure_dir
        
        # self.oversampling_ckpt_dir = args.oversampling_ckpt_dir
        # self.oversampling_log_dir = args.oversampling_log_dir
        # self.oversampling_figure_dir = args.oversampling_figure_dir
        # self.weight_balancing_ckpt_dir = args.weight_balancing_ckpt_dir
        # self.weight_balancing_log_dir = args.weight_balancing_log_dir
        # self.weight_figure_dir = args.weight_balancing_figure_dir
        # self.feature_gan_ckpt_dir = args.feature_gan_ckpt_dir
        # self.feature_gan_log_dir = args.feature_gan_log_dir
        # self.lstm_retrain_ckpt_dir = args.lstm_retrain_ckpt_dir
        # self.lstm_retrain_log_dir = args.lstm_retrain_log_dir
        # self.lstm_retrain_figure_dir = args.lstm_retrain_figure_dir

        self.sequence_length = args.sequence_length
        self.num_classes = args.num_classes
        self.network = args.network
        self.train_continue = args.train_continue
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.msrc_class = ['Start','Crouch','Navigate','Goggles','Wind up','Shoot','Take a bow','Throw','Protest','Change weapon','Move up','Kick']
        
    
    def gan_train(self):
        
        train_dataset = human_Dataset(data_dir=self.train_data_dir, 
                                      image_size= self.sequence_length)
        train_loader = DataLoader(dataset=train_dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)
        
        fn_pred = lambda output: torch.softmax(output,dim=1)
        fn_pred_ = lambda pred: pred.max(dim=1)[1]
        fn_acc = lambda pred, label:((pred.max(dim=1)[1] == label).type(torch.float)).mean()
        
        if self.network == 'ACGAN':
            netG = ACGAN(in_channels=100, out_channels=3, nker=self.sequence_length, num_classes=self.num_classes).to(self.device)
            netD = Discriminator(in_channels=3, out_channels=1, nker=self.sequence_length, num_classes=self.num_classes).to(self.device)
            
            init_weights(netG, init_type='normal', init_gain=0.02)
            init_weights(netD, init_type='normal', init_gain=0.02)
            
            gen_writer = SummaryWriter(log_dir=os.path.join(self.ACGAN_log_dir, 'generator'))
            dis_writer = SummaryWriter(log_dir=os.path.join(self.ACGAN_log_dir, 'discriminator'))
        

        BCE_criterion = nn.BCELoss().to(self.device)
        CE_criterion = nn.CrossEntropyLoss().to(self.device)
        
        optimG = torch.optim.Adam(netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimD = torch.optim.Adam(netD.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        for epoch in range(self.num_epoch):
            
            loss_D_real_train = []
            loss_D_fake_train = []
            
            G_loss = []
            real_accuracy = []
            fake_accuracy = []
            
            for i, (input_value, labels) in enumerate(train_loader):
                
                input_value = input_value.to(self.device,dtype=torch.float32)
                labels = labels.to(self.device)
                
                set_requires_grad(netD, True)
                real_bce_outputs, real_ce_outputs = netD(input_value)
                
                d_real_loss = BCE_criterion(real_bce_outputs, torch.ones(labels.shape[0], 1).to(self.device)) + CE_criterion(real_ce_outputs, labels)
                
                z = Variable(torch.randn(labels.shape[0], 100).to(self.device))
                
                fake_feature = netG(z, labels)
            
                fake_bce_outputs, fake_ce_outputs = netD(fake_feature.detach())
                
                d_fake_loss = BCE_criterion(fake_bce_outputs, torch.zeros(labels.shape[0], 1).to(self.device)) + CE_criterion(fake_ce_outputs, labels)
                
                d_loss = (d_real_loss + d_fake_loss)/2
                
                loss_D_real_train+=[d_real_loss.item()]
                loss_D_fake_train+=[d_fake_loss.item()]
                
                optimD.zero_grad()
                d_loss.backward()
                optimD.step()
                
                set_requires_grad(netD, False)

                bce_outputs, ce_outputs = netD(fake_feature)
                
                
                real_pred = fn_pred(real_ce_outputs)   
                real_acc = fn_acc(real_pred,labels)
                fake_pred = fn_pred(fake_ce_outputs)   
                fake_acc = fn_acc(fake_pred,labels)
                
                real_accuracy+=[real_acc.item()]
                fake_accuracy+=[fake_acc.item()]
                labels = labels.unsqueeze(-1)
                
                g_loss = BCE_criterion(bce_outputs, labels.to(dtype=torch.float32))+CE_criterion(ce_outputs, labels.squeeze(-1))
                
                G_loss+=[g_loss.item()]
                
                optimG.zero_grad()
                g_loss.backward()
                optimG.step()
                
            
            print('Epoch [{}/{}]\tG_loss : {:.4f}\tD_real_loss : {:.4f}, D_fake_loss : {:.4f}\treal_Accuracy: {:.4f}, fake_Accuracy: {:.4f}' .format(epoch+1, self.num_epoch, np.mean(G_loss) , np.mean(loss_D_real_train), np.mean(loss_D_fake_train), np.mean(real_accuracy), np.mean(fake_accuracy)))       
                
            gen_writer.add_scalar('Generator_Loss', np.mean(G_loss), epoch)
            dis_writer.add_scalar('Discriminator_real_Loss', np.mean(loss_D_real_train), epoch)
            dis_writer.add_scalar('Discriminator_fake_Loss', np.mean(loss_D_fake_train), epoch)
            dis_writer.add_scalar('real_Accuracy', np.mean(real_accuracy), epoch)
            dis_writer.add_scalar('fake_Accuracy', np.mean(fake_accuracy), epoch)
            
            save_model(ckpt_dir=os.path.join(self.ACGAN_ckpt_dir, 'generator'), net=netG, optim=optimG, epoch=1)
            save_model(ckpt_dir=os.path.join(self.ACGAN_ckpt_dir, 'discriminator'), net=netD, optim=optimD, epoch=1)
            
        gen_writer.close()
        dis_writer.close()
    
    def test(self):
        
        if self.network == 'ACGAN':
            netG = ACGAN(in_channels=100, out_channels=3, nker=self.sequence_length, num_classes=self.num_classes).to(self.device)
            optimG = torch.optim.Adam(netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
            netG, _ = load_model(ckpt_dir=os.path.join(self.ACGAN_ckpt_dir,'generator'), net=netG, optim=optimG)
        
        netG.eval()
        labels = [i for i in range(self.num_classes)]
        labels = torch.tensor(labels).to(device=self.device)
        z = Variable(torch.randn(labels.shape[0], 100).to(self.device))
        
        fake_feature = netG(z, labels)
        
        fake_feature = fake_feature.to(device='cpu').detach().numpy()
        fake_feature = np.transpose(fake_feature, (0, 2, 1, 3))
        
        motion_sequences = image_to_motion_sequence(fake_feature, self.sequence_length)
        make_gif_file(motion_sequences, self.sequence_length,  self.ACGAN_figure_dir)
                
        
