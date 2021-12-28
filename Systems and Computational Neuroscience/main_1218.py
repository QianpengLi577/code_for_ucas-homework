# -*- coding: utf-8 -*-
# @Time    : 2021/12/18 10:07
# @Author  : Qianpeng Li
# @FileName: main_1218.py
# @Contact : liqianpeng2021@ia.ac.cn

from torchvision import transforms
#  MNIST data
from torchvision.datasets import MNIST

mean, std = 0.1307, 0.3081

train_dataset = MNIST('../data/MNIST', train=True, download=True,
                      transform=transforms.ToTensor())
test_dataset = MNIST('../data/MNIST', train=False, download=True,
                     transform=transforms.ToTensor())
n_classes = 10

import torch
from torch.optim import lr_scheduler
import torch.optim as optim

from trainer import fit
import numpy as np

cuda = torch.cuda.is_available()


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


# siamese part

batch_size = 200
from datasets import SiameseMNIST

siamese_train_dataset = SiameseMNIST(train_dataset)  # Returns pairs of images and target same/different
siamese_test_dataset = SiameseMNIST(test_dataset)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

from networks import EmbeddingNet, SiameseNet
from losses import ContrastiveLoss

margin = 1.
embedding_net = EmbeddingNet()
model_siamese = torch.load('E:/UCAS/课程/系统与神经计算科学/code_siamese/ckpt0.t7')
# model_siamese = SiameseNet(embedding_net)
if cuda:
    model_siamese.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model_siamese.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 5

# %%

# fit(siamese_train_loader, siamese_test_loader, model_siamese, loss_fn, optimizer, scheduler, n_epochs, cuda,
#     log_interval, name='siamese')

# softmax

from networks import ClassificationNet
from metrics import AccumulatedAccuracyMetric

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

model = ClassificationNet(model_siamese, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.MSELoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 5
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
#     metrics=[AccumulatedAccuracyMetric()], name='softmax')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import time
correct = 0
total = 0
for epoch in range(n_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()

        images = images.float().to(device)
        outputs = model(images)
        labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.cpu().max(1)
        total += float(labels.size(0))
        correct += float(predicted.eq(labels).sum().item())
        if (i+1)%10 == 0:
             print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %(epoch+1, n_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
             acc = 100. * float(correct) / float(total)
             print(i, len(test_loader), ' Acc: %.5f' % acc)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, 0.001, 40)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx %100 ==0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

    print('Iters:', epoch,'\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)

