from shocks.dataset import *
import json
import pickle

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shocks.dataset import BaseDataset


LOBSTER_RESAMPLE_METHODS = {
    "date": "",
    "psell1": "last",
    "vsell1": "last",
    "pbuy1": "last",
    "vbuy1": "last",
    "psell2": "last",
    "vsell2": "last",
    "pbuy2": "last",
    "vbuy2": "last",
    "psell3": "last",
    "vsell3": "last",
    "pbuy3": "last",
    "vbuy3": "last",
    "psell4": "last",
    "vsell4": "last",
    "pbuy4": "last",
    "vbuy4": "last",
    "psell5": "last",
    "vsell5": "last",
    "pbuy5": "last",
    "vbuy5": "last",
    "psell6": "last",
    "vsell6": "last",
    "pbuy6": "last",
    "vbuy6": "last",
    "psell7": "last",
    "vsell7": "last",
    "pbuy7": "last",
    "vbuy7": "last",
    "psell8": "last",
    "vsell8": "last",
    "pbuy8": "last",
    "vbuy8": "last",
    "psell9": "last",
    "vsell9": "last",
    "pbuy9": "last",
    "vbuy9": "last",
    "psell10": "last",
    "vsell10": "last",
    "pbuy10": "last",
    "vbuy10": "last",
    "time": "",
    "event_type": "",
    "order_id": "",
    "size": "sum",
    "price": "last",
    "direction": "last",
    "unk": "",
}


class LobsterDataset(BaseDataset):
    def __init__(self, data):
        super(LobsterDataset, self).__init__()
        self.tmp_data = data

    def preprocess(self, data, freq="30s") -> pd.DataFrame:
        data["price"] = data["price"].ffill()
        data["returns"] = data["price"].pct_change().dropna()
        data["log_returns"] = np.log(data["price"] / data["price"].shift(1)).dropna()
        data["moving_average"] = data["price"].rolling(25).mean()
        return data.dropna()


if __name__ == "__main__":
    symbol = "KO"
    start_date = "2021-11-01"
    end_date = "2022-01-01"
    freq = "30s"
    std_from_mean = 2

    #dataset = LobsterDataset(symbol)
    #with open('/Users/emamerca/dev/shocks/data/lobster/GME_2020-11-02_2021-04-30.pkl', 'rb') as f:
    #    dataset = pickle.load(f)
    #dataset.rename(columns={"return": "returns",
    #                        "log_return": "log_returns",
    #                        "price_mov_avg": "moving_average"},
    #               inplace=True)
    #dataset = LobsterDataset(dataset)
    #data, shocks = dataset.build_dataset(
    #    dir_path=dataset.tmp_data,
    #    start_date=start_date,
    #    end_date=end_date,
    #    freq=freq,
    #    shocks_window=300,
    #    fit_window=300,
    #    std_from_mean=std_from_mean,
    #)
    with open(
        f"/Users/emamerca/dev/shocks/data/featurized/levy_features__{symbol}__{start_date}_{end_date}__{freq}__{std_from_mean}_std.pkl",
        "rb",
    ) as file:
        d = pickle.load(file)
    features = Features(d["data"], d["shocks"])
    f = features.compute(
        pre_shock_offset=5, post_shock_offset=5, feature_offsets=[5, 10, 50, 100]
    )
    with open(
        f"/Users/emamerca/dev/shocks/data/featurized/features__{symbol}__{start_date}_{end_date}__{freq}__{std_from_mean}_std.json",
        "w",
    ) as file:
        json.dump(f, file)



