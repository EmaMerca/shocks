import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shocks.dataset import BaseDataset


LOBSTER_RESAMPLE_METHODS = {
    "date": "",
    "psell1": "mean",
    "vsell1": "sum",
    "pbuy1": "mean",
    "vbuy1": "sum",
    "psell2": "mean",
    "vsell2": "sum",
    "pbuy2": "mean",
    "vbuy2": "sum",
    "psell3": "mean",
    "vsell3": "sum",
    "pbuy3": "mean",
    "vbuy3": "sum",
    "psell4": "mean",
    "vsell4": "sum",
    "pbuy4": "mean",
    "vbuy4": "sum",
    "psell5": "mean",
    "vsell5": "sum",
    "pbuy5": "mean",
    "vbuy5": "sum",
    "psell6": "mean",
    "vsell6": "sum",
    "pbuy6": "mean",
    "vbuy6": "sum",
    "psell7": "mean",
    "vsell7": "sum",
    "pbuy7": "mean",
    "vbuy7": "sum",
    "psell8": "mean",
    "vsell8": "sum",
    "pbuy8": "mean",
    "vbuy8": "sum",
    "psell9": "mean",
    "vsell9": "sum",
    "pbuy9": "mean",
    "vbuy9": "sum",
    "psell10": "mean",
    "vsell10": "sum",
    "pbuy10": "mean",
    "vbuy10": "sum",
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
            a = 0
            df.drop(0, axis=0, inplace=True)
        df.index = pd.to_datetime(df["date"])
        df.drop(["date", "time", "event_type", "order_id", "unk"], axis=1, inplace=True)
        price_columns = [
            col
            for col in df.columns
            if col == "price" or "psell" in col or "pbuy" in col
        ]
        df = df.apply(pd.to_numeric)
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
        return data.dropna()
