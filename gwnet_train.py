import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from models import gwnet, rmse_loss
from Dataset import BikeDataSet
import scipy.sparse as sp
from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, gcn_bool,
                 addaptadj, aptinit):
        self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj,
                           aptinit=aptinit, in_dim=in_dim, out_dim=1, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16,
                           kernel_size=4, blocks=8, layers=2)
        self.scaler = scaler
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = rmse_loss
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        predict = self.model(input)
        loss = self.loss(predict, real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return loss.item()

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val)
        return loss.item()


if __name__ == '__main__':
    device = torch.device('cuda:1')
    scaler = StandardScaler()
    dataset = BikeDataSet(scaler)
    dataloader = DataLoader(dataset, num_workers=1, batch_size=48)
    engine = trainer(scaler=scaler, in_dim=1, seq_length=24, num_nodes=308, nhid=32, dropout=0.3, lrate=0.01,
                     wdecay=0.0001, device=device, supports=None, gcn_bool=True, addaptadj=True, aptinit=None)
    best_loss = 10000
    best_epoch = -1
    epoch_list = []
    for epoch in range(1000):
        iter_loss_list = []
        for iter, (x, y) in enumerate(dataloader):
            x = x.float().permute(0, 2, 1).unsqueeze(1).to(device)
            y = y.float().to(device)
            l = engine.train(x, y)
            l = l * np.sqrt(scaler.var_)
            iter_loss_list.append(l)

        epoch_loss = np.mean(iter_loss_list)
        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
        elif best_epoch + 20 < epoch:
            break
        print("Epoch:", epoch, "loss:", epoch_loss, "Best Epoch:", best_epoch, "Best Loss:", best_loss)
