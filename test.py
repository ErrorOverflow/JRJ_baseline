import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import mse_loss


class BikeDataSet(Dataset):
    def __init__(self, predict_length=13):
        self.path = "/data/DiskData/ecard-xicheng/up2.csv"
        df = pd.read_csv(self.path, index_col=0)
        self.data = df.values
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        self.predict_length = predict_length

    def __getitem__(self, index):
        return self.data[index:index + self.predict_length, :], self.data[index + self.predict_length + 1, :]

    def __len__(self):
        return len(self.data) - self.predict_length - 1


class GRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GRU = torch.nn.GRU(input_size=345, hidden_size=345)

    def forward(self, data):
        x = data.permute(1, 0, 2).contiguous()
        res = self.GRU(x)
        return res


def rmse_loss(pred, real_val):
    return torch.sqrt(mse_loss(pred, real_val))


class Trainer():
    def __init__(self, device, lrate=0.01, wdecay=0.0001):
        self.model = GRU().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        pass

    def train(self, input, real_val):
        _, output = self.model(input)
        predict = torch.squeeze(output)
        loss = rmse_loss(predict, real_val)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval(self):
        pass


if __name__ == '__main__':
    device = torch.device('cuda')
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


