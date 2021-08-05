import pandas as pd
from torch.utils.data import Dataset


class BikeDataSet(Dataset):
    def __init__(self, scaler, predict_length=24):
        df_loc = pd.read_excel("/data/DiskData/ecard-xicheng/北京_交通设施服务.xls")
        self.path = "/data/DiskData/ecard-xicheng/up.csv"
        df = pd.read_csv(self.path, index_col=0)
        drop_list = []
        for station_name in df.columns:
            if station_name not in list(df_loc['name']):
                drop_list.append(station_name)
        df = df.drop(columns=drop_list)
        self.data = df.values
        self.scaler = scaler
        self.data = scaler.fit_transform(self.data)
        self.predict_length = predict_length

    def __getitem__(self, index):
        return self.data[index:index + self.predict_length, :], self.data[index + self.predict_length + 1, :]

    def __len__(self):
        return len(self.data) - self.predict_length - 1
