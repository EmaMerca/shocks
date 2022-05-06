import os

import pandas as pd
import datetime
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import json

import pystable


from shocks import *

__all__ = ["FREQS", "COLUMNS", "Dataset"]


FREQS = {
    "1m": "1T",
    "5m": "5T",
    "10m": "10T",
    "30m": "30T",
    "1h": "60T",
    "12h": "720T",
    "1d": "1440T",
    "5d": "7200T",
    "10d": "14400T",
}

COLUMNS = (
    "open time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close time",
    "quote asset volume",
    "number of trades",
    "taker buy base asset volume",
    "taker buy quote asset volume",
    "ignore",
)


@pd.api.extensions.register_series_accessor("tot_pct_change")
class TotPctChange:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def tot_pct_change(self):
        return (
            self._obj.__getitem__(-1) - self._obj.__getitem__(0)
        ) / self._obj.__getitem__(0)


class Dataset(object):
    def __init__(
        self,
        pair: str,
        cols: list = COLUMNS,
    ):
        self.pair = pair
        self.__columns = cols
        self.data = self._load_data()
        self.raw_data = self.data.copy()
        self.shocks = []

    def _load_data(self) -> pd.DataFrame:
        pair_dir = from_root("data", "binance", self.pair)
        df_list = [
            pd.read_csv(fname, names=self.__columns)
            for fname in [
                os.path.join(pair_dir, file) for file in sorted(os.listdir(pair_dir))
            ]
        ]
        return pd.concat(df_list, axis=0)

    @staticmethod
    def resample_data(
        df: pd.DataFrame, freq: str = "1h", mode: str = "last"
    ) -> pd.DataFrame:
        ohlc_dict = {"close": mode, "volume": "sum"}
        return (
            df.resample(FREQS[freq], closed="left", label="left")
            .apply(ohlc_dict)
            .dropna()
        )

    @staticmethod
    def filter_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (df.index > start_date) & (df.index < end_date)
        return df.loc[mask]

    @staticmethod
    def datetime_index(
        df: pd.DataFrame, time_col: str = "open time", is_millisec: bool = True
    ) -> pd.DataFrame:
        def to_human_time(unix_time):
            if is_millisec:
                unix_time /= 1000
            t = datetime.datetime.utcfromtimestamp(unix_time)
            return f"{t.year}-{t.month}-{t.day} {t.hour}:{t.minute}"

        df.index = pd.to_datetime(df[time_col].apply(to_human_time))
        return df

    def preprocess(self, freq="1h", keep_cols: List[str] = ("close", "volume")) -> None:
        self.data = Dataset.resample_data(
            df=Dataset.datetime_index(df=self.raw_data), freq=freq
        )
        assert self.data.columns.tolist() == list(
            keep_cols
        ), "columns are missing, consider adding them in the ohlc_dict in the resample method"
        for col in keep_cols:
            self.data[col] = self.data[col].ffill()
        self.data["close"] = self.data["close"].ffill()
        self.data["returns"] = self.data["close"].pct_change().dropna()
        # log returns: shift 1 for as ascending (=forward in time), -1 for descending
        self.data["log_returns"] = np.log(
            self.data["close"] / self.data["close"].shift(1)
        ).dropna()

    def find_shocks(
        self,
        start_date: str,
        end_date: str,
        std_from_mean: float = 3.5,
        plot: bool = False,
    ):
        df = Dataset.filter_data(self.data, start_date, end_date)
        # find dates where returns distribution is more than std_from_mean away from mean
        upper_threshold = df["returns"].mean() + std_from_mean * df["returns"].std()
        lower_threshold = df["returns"].mean() - std_from_mean * df["returns"].std()
        shock_dates = df["returns"][
            (df["returns"] <= lower_threshold) | (df["returns"] >= upper_threshold)
        ]

        # shocks is just a list of dates, we need to extract the individual shocks from there
        start = shock_dates.index[0]
        for i in range(len(shock_dates.index) - 1):
            next_start = shock_dates.index[i + 1]
            end = shock_dates.index[i]
            self.shocks.append(
                {
                    "start": start,
                    "end": end,
                }
            )
            start = next_start

        # last shock
        self.shocks.append(
            {
                "start": start,
                "end": shock_dates.index[-1],
            }
        )

        # TODO: refactor in a separate method
        if plot:
            fig, axs = plt.subplots(2)
            fig.suptitle(self.pair)

            for idx, key in enumerate("close returns".split()):
                axs[idx].plot(df[key])
                axs[idx].set_title(key)
                axs[idx].grid(True)

                for i, shock in enumerate(self.shocks):
                    axs[idx].axvline(shock["start"], alpha=0.5, color="red")
                    axs[idx].axvline(shock["end"], alpha=0.5, color="red")
                    label_position = (
                        0.5 if i % 2 == 0 else -0.5
                    )  # alternate labels up/down
                    # TODO: allow annotations
                    # plt.text(shock["start"], label_position, shock["duration"])

    def fit(self, window=250, start_date=None, end_date=None):
        if start_date is not None and end_date is not None:
            df = Dataset.filter_data(self.data, start_date, end_date)
        else:
            df = self.data
        init_fit = {"alpha": 2, "beta": 0, "sigma": 1, "mu": 0, "parameterization": 1}
        dist = pystable.create(
            init_fit["alpha"],
            init_fit["beta"],
            init_fit["sigma"],
            init_fit["mu"],
            init_fit["parameterization"],
        )

        # TODO: make this more efficient: the fit_levy method should only be called once https://stackoverflow.com/questions/22218438/returning-two-values-from-pandas-rolling-apply
        for param in "alpha beta".split():
            df[param] = (
                df["log_returns"]
                .rolling(window)
                .apply(lambda x: self.fit_levy(dist, x, param))
            )

        self.fitted = df.dropna()

    def fit_levy(self, dist, log_returns, return_param):
        pystable.fit(dist, log_returns, len(log_returns))
        return getattr(dist.contents, return_param)

    @staticmethod
    def plot_fit(df, shocks=None):
        fig, axs = plt.subplots(4)
        fig.suptitle("BTC/USD")
        df = df.dropna()
        for idx, key in enumerate("returns close alpha beta".split()):
            axs[idx].plot(df[key])
            axs[idx].set_title(key)
            axs[idx].grid(True)

            if shocks is not None:
                # trans = axs.get_xaxis_transform()
                for i, shock in enumerate(shocks):
                    axs[idx].axvline(shock["start"], alpha=0.5, color="red")
                    axs[idx].axvline(shock["end"], alpha=0.5, color="red")
                    # label_position = 0.5 if i % 2 == 0 else -0.5
                    # plt.text(shock["start"], label_position, shock["duration"])
        plt.show()

    def build_dataset(self, shocks_window=1000, fit_window=250, std_from_mean=3.5):
        def compute_feature(df, shock_idx, idx_before_shock, col, *args):
            """*args are df class methods which can be used with getattr()"""
            df_slice = df.iloc[shock_idx - idx_before_shock - 1 : shock_idx][col]
            for arg in args:
                # in order to use the method "tot_pct_change" we must call df.tot_pct_change.tot_pct_change()
                if arg == "tot_pct_change":
                    df_slice = getattr(df_slice, arg)
                df_slice = getattr(df_slice, arg)()
            return df_slice

        # 1. detect shocks for all data
        for start in range(len(self.data) - shocks_window):
            sliced = self.data.iloc[start : start + shocks_window]
            self.find_shocks(
                start_date=sliced.index[0],
                end_date=sliced.index[-1],
                std_from_mean=std_from_mean,
            )

        # 2. fit data
        self.fit(window=fit_window)

        # 3. create features:
        # avg pct change in alpha, beta, price, volume  at 5, 10, 50 observations before shock
        cols = ("alpha", "beta", "close", "volume")
        observations_before_shocks = (5, 10, 25, 50)
        measures = ("mean", "std", "tot_pct_change")
        operations = "pct_change"

        features = list(product(cols, observations_before_shocks, measures))
        for i in range(len(features)):
            for op in [operations]:
                new = list(features[i])
                if new[-1] != "tot_pct_change":
                    new.insert(2, op)
                    features.append(new)

        times = self.fitted.index.tolist()
        starting_time = self.fitted.index[0]
        shock_features = []
        for shock in self.shocks:
            if shock["start"] <= starting_time:
                continue
            # shock signal should fire off 5 observations before shock happens else it's too late
            shock_index = times.index(shock["start"]) - 5
            shock_features.append(
                {
                    "_".join(map(str, feature)): compute_feature(
                        self.fitted,
                        shock_index,
                        feature[1],
                        feature[0],
                        *feature[2:],
                    )
                    for feature in features
                }
            )

        save_path = from_root("data", "processed")
        with open(f"{save_path}/{self.pair}.json", "w") as f:
            json.dump(
                {
                    "features": shock_features,
                    "data": self.fitted,
                    "shocks": self.shocks,
                },
                f,
            )
        print(f"{self.pair} has been processed succesfully")
