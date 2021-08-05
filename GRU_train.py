import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from models import GRU, rmse_loss
from Dataset import BikeDataSet



class Trainer():
    def __init__(self, device, lrate=0.01, wdecay=0.0001):
        self.model = GRU().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        pass

    def train(self, input, real_val):
        self.optimizer.zero_grad()
        _, output = self.model(input)
        predict = torch.squeeze(output)
        loss = rmse_loss(predict, real_val)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval(self):
        pass


if __name__ == '__main__':
    device = torch.device('cuda:1')
    dataset = BikeDataSet()
    dataloader = DataLoader(dataset, num_workers=1, batch_size=64)
    trainer = Trainer(device)
    best_loss = 10000
    best_epoch = -1
    epoch_list = []
    for epoch in range(1000):
        iter_loss_list = []
        for iter, (x, y) in enumerate(dataloader):
            x = x.float().to(device)
            y = y.float().to(device)
            l = trainer.train(x, y)
            iter_loss_list.append(l)

        epoch_loss = np.mean(iter_loss_list)
        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
        elif best_epoch + 50 < epoch:
            break
        print("Epoch:", epoch, "loss:", epoch_loss, "Best Epoch:", best_epoch, "Best Loss:", best_loss)
