import math
import torch.optim as optim
import models
import dataLoader.dataLoader
import torch
import torch.nn as nn

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1.0, start_decay_at=None):
        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        for param in self.params:
            grad_norm += math.pow(param.grad.data.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:
            shrinkage = 1.

        for param in self.params:
            if shrinkage < 1:
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if validation performance does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is None or epoch <= self.start_decay_at:
            return
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
        #             print("Decaying learning rate to %g" % self.lr)
        # only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad();
        output = model(X);
        scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.item();
        n_samples += (output.size(0) * data.m);
    return total_loss / n_samples


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis=0);
    sigma_g = (Ytest).std(axis=0);
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation, predict

data_path = "/home/wangmulan/Documents/result/"
data = dataLoader.dataLoader.DataLoader(data_path, windows=24 * 7, train_rate=0.6, valid_rate=0.2, horizon=12)
print(data.train[0].shape, data.train[1].shape)  # torch.Size([10347, 168, 1]) torch.Size([10347, 1])
window = data.train[0].shape[1]
n_val = data.train[0].shape[2]

model = models.Model(n_val, window, 32);

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

criterion = nn.MSELoss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')
evaluateL1 = nn.L1Loss(reduction='sum')
criterion = criterion.cuda()
evaluateL1 = evaluateL1.cuda()
evaluateL2 = evaluateL2.cuda()

optimizer = Optim(
    model.parameters(), 'adam', lr=0.01, max_grad_norm=10, start_decay_at=10, lr_decay=0.9
)

batch_size = 128
epochs = 10
best_val = 10000000
save = 'model.pt'

print('begin training')
import time

for epoch in range(1, epochs):
    epoch_start_time = time.time()
    train_loss = train(data, data.train[0], data.train[1], model, criterion, optimizer, batch_size)
    val_loss, val_rae, val_corr, _ = evaluate(data, data.valid[0], data.valid[1], model, evaluateL2, evaluateL1,
                                              batch_size);
    print(
        '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | lr {:5.4f}'
            .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, optimizer.lr))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val:
        with open(save, 'wb') as f:
            torch.save(model, f)
        best_val = val_loss

    if epoch % 5 == 0:
        test_acc, test_rae, test_corr, _ = evaluate(data, data.test[0], data.test[1], model, evaluateL2, evaluateL1,
                                                    batch_size);
        print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

    optimizer.updateLearningRate(val_loss, epoch)
