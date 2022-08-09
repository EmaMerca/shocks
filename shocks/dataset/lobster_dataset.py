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
    def __init__(self, name: str, columns: list = LOBSTER_RESAMPLE_METHODS.keys()):
        super(LobsterDataset, self).__init__()
        self.cols = columns
        self.name = name

    def load_data(self, fname: str) -> pd.DataFrame:
        df = pd.read_csv(fname, names=self.cols)
        # duplicated columns names
        if set(df.columns) == set(df.iloc[0].values):
            df.drop(0, axis=0, inplace=True)
        df.index = pd.to_datetime(df["date"])
        df.drop(["date", "time", "event_type", "order_id", "unk"], axis=1, inplace=True)
        price_columns = [
            col
            for col in df.columns
            if col == "price" or "psell" in col or "pbuy" in col
        ]
        volume_columns = [
            col
            for col in df.columns
            if col == "size" or "vsell" in col or "vbuy" in col
        ]
        df = df.apply(pd.to_numeric)
        df[price_columns].fillna(method="ffill", inplace=True)
        df["direction"].fillna(method="ffill", inplace=True)
        df[volume_columns].fillna(value=0, inplace=True)
        df[price_columns] = df[price_columns].apply(lambda x: x / 10000)
        return df

    @staticmethod
    def resample_data(
        df: pd.DataFrame,
        freq: str = "5m",
        resampling_modes: dict = LOBSTER_RESAMPLE_METHODS,
    ) -> pd.DataFrame:
        resampling_modes = {
            k: v for k, v in resampling_modes.items() if k in df.columns
        }
        return (
            df.resample(LobsterDataset.FREQS[freq], closed="left", label="left")
            .apply(resampling_modes)
            .dropna()
        )

    def preprocess(self, fname: str, freq="1h") -> pd.DataFrame:
        data = LobsterDataset.resample_data(df=self.load_data(fname), freq=freq)
        data["price"] = data["price"].ffill()
        data["returns"] = data["price"].pct_change().dropna()
        data["log_returns"] = np.log(data["price"] / data["price"].shift(1)).dropna()
        data["moving_average"] = data["price"].rolling(25).mean()
        return data.dropna()
