import torch
import util
import models
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/XiAn_City', help='data path')
parser.add_argument('--adjdata', type=str, default='data/XiAn_City/adj_mat.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--seq_length', type=int, default=12, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=792, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--force', type=str, default=False, help="remove params dir", required=False)
parser.add_argument('--save', type=str, default='./garage/XiAn_City', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--model', type=str, default='GRU', help='adj type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')

args = parser.parse_args()


class GRU_trainer():
    def __init__(self, num_nodes, dropout, learning_rate, weight_decay, device, decay):
        self.model = models.GRU(device, num_nodes, dropout)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse


def main():
    device = torch.device(args.device)
    engine = GRU_trainer(args.num_nodes, args.dropout,
                         args.learning_rate, args.weight_decay, device, args.decay)


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
