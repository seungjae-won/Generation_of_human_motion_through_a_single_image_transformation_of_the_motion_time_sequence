from os.path import splitext
from os import listdir
import os
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import sys
from random import sample
import math
import re
import random
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from torch.autograd import Variable
from model import *


class human_Dataset(Dataset):
    
    def __init__(self, data_dir, image_size):
        self.data_file_list = getFileNames(data_dir)
        self.data_dir = data_dir
        self.data,self.label = make_change(readDataFile(self.data_dir,self.data_file_list)), parseFileName(self.data_file_list)
    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])

def getFileNames(input_path):
    
    input_path = glob.glob('%s/*.csv' % input_path)
    gestureId_list = [[] for _ in range(12)]
    
    for i in range(len(input_path)):
        input_path[i] = input_path[i].split("/")[-1]
        noSuffix = input_path[i].split(".")[0]
        fields = noSuffix.split("_")	
        gestureId = fields[2]				
        if (gestureId[-1] == 'A'):
            twoModalities = True
            gestureId = gestureId[:-1]
        gestureId = int(gestureId)
        gestureId_list[gestureId-1].append(input_path[i])
    
    csv_list = []

    for i in range(len(gestureId_list)):
        for j in range(len(gestureId_list[i])):
            csv_list.append(gestureId_list[i][j])
    
    return csv_list

def parseFileName(data_list):
    
    label_list = []
    
    for fName in data_list:
        fName = fName.split("/")[-1]
        noSuffix = fName.split(".")[0]
        fields = noSuffix.split("_")	
        gestureId = fields[2]				
        if (gestureId[-1] == 'A'):
            twoModalities = True
            gestureId = gestureId[:-1]

        label = (int(gestureId)-1)
        
        label_list.append(label)

    return label_list

def readDataFile(data_dir, dataFile, image_size):
    
    all_data_list = []
    for fName in dataFile:
        
        data_path = os.path.join(data_dir,fName)
        contents = np.genfromtxt(data_path, delimiter=' ')
        data = list(contents[:,1:])

        data_list = []

        for i in range(len(data)):
            data_line = []
            
            if data[i][0] == 0 and data[i][1] == 0 and data[i][2] == 0:
                continue 
            while len(data[i]) != 0:
                for _ in range(3):
                    data_line.append(data[i][0])
                    data[i] = np.delete(data[i],0)
                data[i] = np.delete(data[i],0)
            data_list.append(data_line)

        index = []
        A = len(data_list) // image_size
        
        for i in range(0,len(data_list),A):
            if len(index) == image_size:
                break
            
            check = []
            for j in range(60):
                sum=0
                for k in range(i,i+A):
                    sum+=data_list[k][j]
                check.append(sum/A)
                
            index.append(check)
        data_list = index
        
        all_data_list.append(data_list)
    
    
    all_data_list = np.array(all_data_list)

    return all_data_list


def make_change(data_list):
    to_image = []

    for j in range(len(data_list)):      
        sequence = []
        for k in range(len(data_list[j])):
            x = []
            y = []
            z = []
            for l in range(len(data_list[j][k])):
                if l % 3 == 0:
                    for _ in range(12):
                        x.append(data_list[j][k][l])
                    if l // 3 != 0 and l // 3 != 1 and l // 3 != 12 and l // 3 != 16:
                        x.append(data_list[j][k][l])
                elif l % 3 ==1:
                    for _ in range(12):
                        y.append(data_list[j][k][l])
                    if l // 3 != 0 and l // 3 != 1 and l // 3 != 12 and l // 3 != 16:
                        y.append(data_list[j][k][l])
                elif l % 3 ==2:
                    for _ in range(12):
                        z.append(data_list[j][k][l])
                    if l // 3 != 0 and l // 3 != 1 and l // 3 != 12 and l // 3 != 16:
                        z.append(data_list[j][k][l])
            sum_ = []
            for c in range(len(x)):
                sum_.append([x[c],y[c],z[c]])
            sequence.append(sum_)
        to_image.append(sequence)

    to_image = np.array(to_image)
    return to_image