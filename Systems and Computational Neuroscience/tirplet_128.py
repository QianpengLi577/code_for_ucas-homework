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
from datasets import SiameseMNIST,TripletMNIST

siamese_train_dataset = TripletMNIST(train_dataset)  # Returns pairs of images and target same/different
siamese_test_dataset = TripletMNIST(test_dataset)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

from networks import EmbeddingNet, SiameseNet ,TripletNet
from losses import ContrastiveLoss ,TripletLoss

margin = 1.
embedding_net = EmbeddingNet()
model_siamese = TripletNet(embedding_net)
# model_siamese = torch.load('E:/UCAS/课程/系统与神经计算科学/code_siamese/CPK/ckpt_siamese_10_epochmax10_9.t7')
if cuda:
    model_siamese.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model_siamese.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 10
log_interval = 5

# %%

fit(siamese_train_loader, siamese_test_loader, model_siamese, loss_fn, optimizer, scheduler, n_epochs, cuda,
    log_interval)

# softmax

from networks import ClassificationNet
from metrics import AccumulatedAccuracyMetric

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

model = ClassificationNet(model_siamese, n_classes=n_classes)
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 60
log_interval = 50
fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    metrics=[AccumulatedAccuracyMetric()] )
