import os

import pandas as pd
import datetime
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pickle
from tqdm import tqdm_notebook
import random

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
        self.freq = freq
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
        def compute_features(data, shock_idx, idx_before_shock, *ops):
            res = data[:, shock_idx - idx_before_shock - 1 : shock_idx]
            for op in ops:
                res = operations[op](res)
            return res

        def assign_name(columns: list, feature_names: list) -> list:
            return list(
                map(
                    lambda x: f"{x}_{'_'.join(map(str, feature_names))}",
                    columns,
                )
            )

        from time import time
        t = time()
        # 1. detect shocks for all data
        for start in range(0, len(self.data) - shocks_window, shocks_window):
            sliced = self.data.iloc[start : start + shocks_window]
            self.find_shocks(
                start_date=sliced.index[0],
                end_date=sliced.index[-1],
                std_from_mean=std_from_mean,
            )
        print("shocks detected. Time: ", time() - t)
        t = time()
        # 2. fit data
        self.fit(window=fit_window)
        print("Data fitted. Time: ", time() - t)
        t = time()
        # 3. create features:
        # avg pct change in alpha, beta, price, volume  at 5, 10, 50 observations before shock
        cols = ["alpha", "beta", "volume", "close"]  # close should be last column
        observations_before_shocks = (5, 10, 25, 50)
        measures = ("mean", "std", "tot_pct_change")
        additional_measures = "pct_change"

        features_names = list(product(observations_before_shocks, measures))
        for i in range(len(features_names)):
            for add_meas in [additional_measures]:
                new = list(features_names[i])
                if new[-1] != "tot_pct_change":
                    new.insert(1, add_meas)
                    features_names.append(new)

        operations = {
            "mean": lambda x: np.mean(x, axis=1),
            "std": lambda x: np.std(x, axis=1),
            "pct_change": lambda x: np.diff(x) / x[:, :-1] * 100,
            "tot_pct_change": lambda x: 100 * (x[:, -1] - x[:, 0]) / x[:, 0],
        }

        times = self.fitted.index.tolist()
        starting_time = self.fitted.index[0]
        shocks_indexes = []
        shock_features = []
        np_data = self.fitted[cols].to_numpy().T
        for shock in tqdm_notebook(self.shocks):
            if shock["start"] <= starting_time:
                continue
            # shock signal should fire off 5 observations before shock happens else it's too late
            shock_index = times.index(shock["start"])
            shocks_indexes.extend((i for i in range(shock_index - 5, shock_index + 5)))
            shock_feature = {}
            for feature_name in features_names:
                # compute 1 feature for all the cols
                feature = compute_features(
                    np_data,
                    shock_index - 5,
                    feature_name[0],
                    *feature_name[1:],
                )
                named_features = dict(
                    zip(
                        assign_name(cols, feature_name),
                        feature,
                    )
                )
                shock_feature = dict(
                    named_features,
                    **shock_feature,
                )
            # -1 if price drops, 1 if price increases
            shock_feature["direction"] = (
                -1 if np_data[-1, shock_index - 1] >= np_data[-1, shock_index] else 1
            )
            shock_features.append(shock_feature)
        print("Shocks processed. Time: ", time() - t)
        t = time()
        # add non shocks samples
        non_shock_indexes = [
            i
            for i in range(observations_before_shocks[-1] + 5 + 1, len(times))
            if i not in shocks_indexes
            and times[i] > starting_time
        ]

        for non_shock_index in tqdm_notebook(
            random.sample(non_shock_indexes, min(len(self.fitted) // 2, 50 * len(self.shocks)))
        ):
            non_shock_feature = {}
            for feature_name in features_names:
                # compute 1 feature for all the cols
                feature = compute_features(
                    np_data,
                    non_shock_index - 5,
                    feature_name[0],
                    *feature_name[1:],
                )
                named_features = dict(
                    zip(
                        assign_name(cols, feature_name),
                        feature,
                    )
                )
                non_shock_feature = dict(
                    named_features,
                    **non_shock_feature,
                )
            # 0 means not a shock
            non_shock_feature["direction"] = 0
            shock_features.append(non_shock_feature)

        print("Non shocks processed. Time: ", time() - t)
        save_path = from_root("data", "processed")
        with open(f"{save_path}/{self.pair}_{self.freq}.pkl", "wb") as f:
            pickle.dump(
                {
                    "features": shock_features,
                    "data": self.fitted,
                    "shocks": self.shocks,
                },
                f,
            )
        print(f"{self.pair} has been processed succesfully")
