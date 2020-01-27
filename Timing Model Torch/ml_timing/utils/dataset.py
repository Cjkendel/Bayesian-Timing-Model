import torch
import pandas as pd
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_file, static_features_count=None, categorical_features=[]):
        df = pd.read_csv(path_to_file)
        self.total_data_len = len(df)
        cols_list = df.columns.to_list()
        static_features_count = static_features_count or cols_list.index('M')
        static_features_cols = cols_list[:static_features_count]

        self.m_series = df['M']

        self.df_static_features = pd.concat(
            [self.__class__.sanitize_static_feature(df[f]) for f in static_features_cols],
            axis=1)
        self.df_send_times = df[cols_list[static_features_count + 1::2]].applymap(self.time_str_to_f)
        self.df_open_times = df[cols_list[static_features_count + 2::2]].applymap(self.time_str_to_f)
        self.add_time_on_negatives()

    @staticmethod
    def sanitize_static_feature(f):
        if not(f.dtype == object or f.dtype == np.int64):
            return f
        return pd.get_dummies(f, prefix=f.name)

    def static_features_number(self):
        return len(self.df_static_features.columns)

    def __getitem__(self, index):
        m = self.m_series[index]
        return (
            torch.tensor(self.df_static_features.iloc[index].to_list()),
            torch.tensor(
                list(zip(
                    self.df_send_times.iloc[index].to_list()[:m],
                    self.df_open_times.iloc[index].to_list()[:m]
                ))
            )
        )

    def __len__(self):
        return self.m_series.size

    @staticmethod
    def time_str_to_f(s):
        if not isinstance(s, str):
            return 0.0
        h, m = [int(x) for x in s.split(':')]
        return h + m / 60.0

    def add_time_on_negatives(self):
        subs = self.df_open_times - self.df_send_times.values
        subs = self.df_open_times + (self.df_open_times[subs < 0] - self.df_open_times[subs < 0] + 24).fillna(0)
        self.df_send_times = subs - self.df_send_times.values

