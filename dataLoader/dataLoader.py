import h5py
import numpy as np
import torch
import normalization
import os


class traffic_demand_prediction_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, key, val_len, test_len):
        self.x = x
        self.y = y
        self.key = key
        self._len = {"train_len": x.shape[0] - val_len - test_len,
                     "validate_len": val_len, "test_len": test_len}

    def __getitem__(self, item):
        if self.key == 'train':
            return self.x[item], self.y[item]
        elif self.key == 'validate':
            return self.x[self._len["train_len"] + item], self.y[self._len["train_len"] + item]
        elif self.key == 'test':
            return self.x[-self._len["test_len"] + item], self.y[-self._len["test_len"] + item]
        else:
            raise NotImplementedError()

    def __len__(self):
        return self._len[f"{self.key}_len"]


class DataLoader():
    def __init__(self, path, windows, horizon, train_rate, valid_rate, normalize="Standard", category="bike"):
        self.window = windows
        self.horizon = horizon
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.normalize = normalize
        self.category = category
        self.data = np.ndarray([0])
        self._read_h5()
        self.n, self.m = self.data[:,:,0].shape
        self.scale = np.ones(self.m)
        self.scale = torch.from_numpy(self.scale).float()
        self._split(int(train_rate * self.n), int((train_rate + valid_rate) * self.n))

    def _read_h5(self):
        normal_method = getattr(normalization, self.normalize)
        with h5py.File(os.path.join(data_path, self.category + "_data.h5"), 'r') as hf:
            data_pick = hf[self.category + "_pick"][:]
            data_drop = hf[self.category + "_drop"][:]
            print(data_pick)
            print(data_pick.shape, data_drop.shape)
            self.data = normal_method().fit_transform(X=np.stack([data_pick, data_drop], axis=2))

        print(self.data.shape)  # (4368, 484, 2)
        # self.data = np.concatenate(self.data, axis=1)

    def _split(self, train_left_flag, valid_left_flag, ):
        train_set = range(self.window + self.horizon - 1, train_left_flag)
        valid_set = range(train_left_flag, valid_left_flag)
        test_set = range(valid_left_flag, self.n)
        self.train = self._batchify(train_set)
        self.valid = self._batchify(valid_set)
        self.test = self._batchify(test_set)

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.window, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.horizon + 1
            start = end - self.window
            X[i, :, :] = torch.from_numpy(self.data[start:end, :])
            Y[i, :] = torch.from_numpy(self.data[idx_set[i], :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X, Y = inputs[excerpt], targets[excerpt]
            X, Y = X.cuda(), Y.cuda()
            yield torch.autograd.Variable(X), torch.autograd.Variable(Y)
            start_idx += batch_size


# def get_data_loader(data_path: str, category: str, Normal_Method: str):
#     data = list()
#
#     normal_method = getattr(normalization, Normal_Method)
#
#     with h5py.File(os.path.join(data_path, category + "_data.h5"), 'r') as hf:
#         data_pick = hf[category + "_pick"][:]
#         data_drop = hf[category + "_drop"][:]
#         print(data_pick)
#         print(data_pick.shape, data_drop.shape)
#         data.append(normal_method().fit_transform(X=np.stack([data_pick, data_drop], axis=2)))
#
#     data = np.concatenate(data, axis=1)
#     print(type(data), data[:, 0, 0].shape)
#     print(data[:, 0, 0])
#     X, Y, X_list, Y_list = [], [], [], []
#
#     for i in range(0, data.shape[0]):
#         X.append([data[i - j] for j in X_list])
#         Y.append([data[i + j] for j in Y_list])


# X_ = torch.from_numpy(np.asarray(X_)).float()
# Y_ = torch.from_numpy(np.asarray(Y_)).float()
# dls = dict()
#
# for key in ['train', 'validate', 'test']:
#     dataset = traffic_demand_prediction_dataset(X_, Y_, key, val_len, test_len)
#     dls[key] = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=16)
# return dls, normal


if __name__ == '__main__':
    data_path = "/home/wangmulan/Documents/result/"
    data = DataLoader(data_path, windows=24 * 7, train_rate=0.6, valid_rate=0.2, horizon=12)
    get_data_loader(data_path, category="bike", Normal_Method="Standard")
