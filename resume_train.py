import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from Model import getModel
from dataset import getDataset
from utils import calc_psnr, AverageMeter


# Connecting to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Connected to {}".format(device))

# Loading the Dataset
train_dataset, val_dataset, test_dataset = getDataset()
print("Dataset has been loaded")

# Create data loaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Loading the model
model = getModel()
model.load_state_dict(torch.load('Saves/Model_Saves/epoch_299.pth'))
model.to(device)
print("Model has been created")

# Loading old arrays
np_psnrs = np.load('Saves/Arrays/psnrs.npy')
np_train_history = np.load('Saves/Arrays/train_history.npy')
np_val_history = np.load('Saves/Arrays/val_history.npy')
psnrs = np_psnrs.tolist()
train_history = np_train_history.tolist()
val_history = np_val_history.tolist()
print("Arrays have been Loaded")


# Training
outputDIR = 'Saves/Model_Saves/'

if not os.path.exists(outputDIR):
    os.makedirs(outputDIR)

cudnn.benchmark = True
torch.manual_seed(42)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

best_weights = copy.deepcopy(model.state_dict())
best_epoch = np_psnrs.argmax()
best_psnr = np_psnrs.max()
num_epochs = 200
batch_size = 4
old_epochs = len(psnrs)

for epoch in range(old_epochs, num_epochs + old_epochs, 1):
    model.train()
    epoch_losses = AverageMeter()

    with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch+1, old_epochs + num_epochs))
        train_loss = 0.0
        for data in train_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))

    torch.save(model.state_dict(), os.path.join(outputDIR, 'epoch_{}.pth'.format(epoch)))

    model.eval()
    epoch_psnr = AverageMeter()
    val_loss = 0
    for data in val_loader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs)
            loss = criterion(preds, labels)
            val_loss += loss.item()
            preds = preds.clamp(0.0, 1.0)

        epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

    print('train loss: {:.6f}'.format(train_loss/len(train_dataset)))
    print('eval loss: {:.6f}'.format(val_loss/len(val_dataset)))
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
    psnrs.append(epoch_psnr.avg.cpu())
    train_history += [train_loss/len(train_dataset)]
    val_history += [val_loss/len(val_dataset)]

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        best_weights = copy.deepcopy(model.state_dict())

print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
torch.save(best_weights, os.path.join(outputDIR,'best_resumed.pth'))

# Plots and array saves
plt.figure()
plt.plot(train_history, 'b',label="trianing loss")
plt.plot(val_history, 'r',label = "validation loss")
plt.title('Convergence plot of gradient descent')
plt.xlabel('Epoch No')
plt.ylabel('J')
plt.legend()
plt.savefig('Saves/Graphs/loss_graph_resumed.jpg')

plt.figure()
plt.plot(psnrs)
plt.title('Convergence plot of PSNR')
plt.xlabel('Epoch No')
plt.ylabel('PSNR')
plt.savefig('Saves/Graphs/psnrs_graph_resumed.jpg')

print("Saved Graphs")

np_psnrs = np.array(psnrs)
np_train_history = np.array(train_history)
np_val_history = np.array(val_history)

np.save('Saves/Arrays/psnrs_resumed.npy', np_psnrs)
np.save('Saves/Arrays/val_history_resumed.npy',np_val_history)
np.save('Saves/Arrays/train_history_resumed.npy',np_train_history)

print("Saved Arrays")