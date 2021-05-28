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
        self.data = list()
        self._read_h5()
        self._split()

    def _read_h5(self):
        normal_method = getattr(normalization, self.normalize)
        with h5py.File(os.path.join(data_path, self.category + "_data.h5"), 'r') as hf:
            data_pick = hf[self.category + "_pick"][:]
            data_drop = hf[self.category + "_drop"][:]
            print(data_pick)
            print(data_pick.shape, data_drop.shape)
            self.data.append(normal_method().fit_transform(X=np.stack([data_pick, data_drop], axis=2)))

        self.data = np.concatenate(self.data, axis=1)

    def _split(self):
        train_set = range(self.window + self.horizon - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)


def get_data_loader(data_path: str, category: str, Normal_Method: str):
    data = list()

    normal_method = getattr(normalization, Normal_Method)

    with h5py.File(os.path.join(data_path, category + "_data.h5"), 'r') as hf:
        data_pick = hf[category + "_pick"][:]
        data_drop = hf[category + "_drop"][:]
        print(data_pick)
        print(data_pick.shape, data_drop.shape)
        data.append(normal_method().fit_transform(X=np.stack([data_pick, data_drop], axis=2)))

    data = np.concatenate(data, axis=1)
    print(type(data), data[:, 0, 0].shape)
    print(data[:, 0, 0])
    X, Y, X_list, Y_list = [], [], [], []

    for i in range(0, data.shape[0]):
        X.append([data[i - j] for j in X_list])
        Y.append([data[i + j] for j in Y_list])


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
    data = DataLoader(data_path, windows=24 * 7, train_rate=0.6, valid_rate=0.2)
    get_data_loader(data_path, category="bike", Normal_Method="Standard")
