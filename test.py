import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt
import numpy as np

from Model import getModel
from dataset import getDataset
from utils import calc_psnr, AverageMeter

if __name__ == '__main__':
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--test-file', type=str, default='Saves/Model_Saves/best.pth')
    args = parser.parse_args()

    # Connecting to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Connected to {}".format(device))
    
    # Loading the Dataset
    train_dataset, val_dataset, test_dataset = getDataset()
    print("Dataset has been loaded")
    
    # Create data loaders for training, validation, and test sets
    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # Loading the model
    model = getModel()
    model.load_state_dict(torch.load(args.test_file))
    model.to(device)
    print("Model has been loaded")

    # Testing Loop
    model.eval()
    criteria = nn.MSELoss()
    epoch_psnr = AverageMeter()
    test_loss = 0
    for data in test_loader:
        inputs, labels = data
    
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        with torch.no_grad():
            preds = model(inputs)
            loss = criteria(preds, labels)
            test_loss += loss.item()
            preds = preds.clamp(0.0, 1.0)
    
        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
    
    print('Test loss: {:.6f}'.format(test_loss/len(test_dataset)))
    print('Test psnr: {:.2f}'.format(epoch_psnr.avg))