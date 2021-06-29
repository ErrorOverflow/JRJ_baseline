import h5py
import numpy as np
import torch
import os


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoader:
    def __init__(self, path, windows, horizon, train_rate, valid_rate, normalize,
                 category="bike"):
        self.data_path = path
        self.window = windows
        self.horizon = horizon
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self._normal_method = normalize
        self.category = category
        self.data = np.ndarray([0])
        self.data_statistics = {"std": 0.0, "mean": 0.0}
        self._read_h5()
        self.n, self.m = self.data.shape
        self.scale = np.ones(self.m)
        self._split(int(train_rate * self.n), int((train_rate + valid_rate) * self.n))
        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.scale = torch.autograd.Variable(self.scale)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _read_h5(self):
        with h5py.File(os.path.join(self.data_path, self.category + "_data.h5"), 'r') as hf:
            data_pick = hf[self.category + "_pick"][:]
            data_drop = hf[self.category + "_drop"][:]
            self.data = self._normal_method.fit_transform(X=np.stack([data_pick, data_drop], axis=2))
            # noinspection PyBroadException
            # try:
            #     self.data_statistics["std"] = self._normal_method.get_std()
            #     self.data_statistics["mean"] = self._normal_method.get_mean()
            # except Exception as E:
            #     print(E)
            #     print("normalization class don't have a get_std() or get_mean() method")

        self.data = self.data[:, :, 0]
        # self.data = np.concatenate(self.data, axis=1)

    def _split(self, train_left_flag, valid_left_flag):
        train_set = range(self.window + self.horizon - 1, train_left_flag)
        valid_set = range(train_left_flag, valid_left_flag)
        test_set = range(valid_left_flag, self.n)
        self.train = self._batch(train_set)
        self.valid = self._batch(valid_set)
        self.test = self._batch(test_set)

    def _batch(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.window, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, :] = torch.from_numpy(self.data[idx_set[i], :])

        return [X, Y]

    @staticmethod
    def get_batches(inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X, Y = inputs[excerpt], targets[excerpt]
            # X, Y = X.cuda(), Y.cuda()
            yield torch.autograd.Variable(X), torch.autograd.Variable(Y)
            start_idx += batch_size


if __name__ == '__main__':
    data_path = "/home/wangmulan/Documents/result/"
    data = DataLoader(data_path, windows=24 * 7, train_rate=0.6, valid_rate=0.2, horizon=12)
    # (data_path, category="bike", Normal_Method="Standard")
