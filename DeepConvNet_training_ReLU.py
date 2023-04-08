import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules import transformer
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import torch.optim as optim
from torchvision import transforms
import pandas as pd
import os
import argparse
from models import DeepConvNetReLU


def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)


    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
   

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

   

    return train_data, train_label, test_data, test_label

def testing(x_test,y_test,model,device):
    model.eval()
    with torch.no_grad():
        model.cuda(0)
        n = x_test.shape[0]

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32").reshape(y_test.shape[0],)

        x_test, y_test = Variable(torch.from_numpy(x_test)),Variable(torch.from_numpy(y_test))
        x_test,y_test = x_test.to(device),y_test.to(device)
        y_pred_test = model(x_test)
        correct = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        # print("testing accuracy:",correct/n)
    return correct/n

train_data, train_label, test_data, test_label = read_bci_data()

n = train_data.shape[0]
# epochs = 3000

device = torch.device("cuda:0")

train_data = train_data.astype("float32")
train_label = train_label.astype("float32").reshape(train_label.shape[0],)


x, y = Variable(torch.from_numpy(train_data)),Variable(torch.from_numpy(train_label))
y=torch.tensor(y, dtype=torch.long) 


    
epochs = 2000
lr = 1e-3
ReLU_max_val_acc = 0
save_model = False
model = DeepConvNetReLU()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 1200], gamma=0.5)

model.to(device)

loss_history = []
ReLU_train_accuracy_history = []
ReLU_test_accuracy_history = []
for epoch in range(epochs):
    # for idx,(data,target) in enumerate(loader):
    model.train()
    x,y = x.to(device),y.to(device)
    y_pred = model(x)

    loss = criterion(y_pred, y)
    train_loss = loss.item()
    loss_history.append(train_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch%1==0:

        # correct= (y_pred.ge(0.5) == y).sum().item()
        n = y.shape[0]
        correct = (torch.max(y_pred,1)[1]==y).sum().item()
        train_accuracy = correct / n
        ReLU_train_accuracy_history.append(train_accuracy)

        # print("epochs:",epoch,"loss:",loss.item(),"Accuracy:",(correct / n),"Learning rate:",scheduler.get_last_lr()[0])
        test_accuracy = testing(test_data,test_label,model,device)
        ReLU_test_accuracy_history.append(test_accuracy)

        print("epochs:",epoch,"loss:",train_loss,"Training Accuracy:",train_accuracy,"Testing Accuracy:",test_accuracy,"Learning rate:",scheduler.get_last_lr()[0])
        
        if test_accuracy > ReLU_max_val_acc:
            ReLU_max_val_acc = test_accuracy
            if save_model:
                torch.save(model.state_dict(), 'DeepConvNetReLU.bin')

print("Max accuracy:",ReLU_max_val_acc)

from matplotlib import pyplot as plt
#plot the training and validation accuracy and loss at each epoch

epochs = range(1, len(ReLU_train_accuracy_history) + 1)
plt.plot(epochs, ReLU_train_accuracy_history, label='ReLU_train', color='r')
plt.plot(epochs, ReLU_test_accuracy_history, label='ReLU_test', color='b')
plt.title('DeepConvNet acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()