from abc import abstractmethod
import os

import pandas as pd
import datetime
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pickle
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor
from numpy.lib.stride_tricks import sliding_window_view

import pystable
from shocks.utils import *


class BaseDataset:
    FREQS = {
        "1s": "1S",
        "5s": "5S",
        "15s": "15S",
        "30s": "30S",
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

    def __init__(self):
        self.data = []
        self.shocks = []

    @abstractmethod
    def preprocess(self, fname: str, freq: str):
        raise NotImplementedError

    @staticmethod
    def filter_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (df.index >= start_date) & (df.index <= end_date)
        return df.loc[mask]

    @staticmethod
    def find_shocks(
        df,
        std_from_mean: float = 3.5,
    ) -> list:
        # find dates where returns distribution is more than std_from_mean away from mean
        df = df[["returns", "log_returns"]]
        upper_threshold = df["returns"].mean() + std_from_mean * df["returns"].std()
        lower_threshold = df["returns"].mean() - std_from_mean * df["returns"].std()
        shock_dates = df["returns"][
            (df["returns"] <= lower_threshold) | (df["returns"] >= upper_threshold)
        ]

        # shocks is just a list of dates, we need to extract the individual shocks from there
        start = shock_dates.index[0]
        shocks = []
        for i in range(len(shock_dates.index) - 1):
            next_start = shock_dates.index[i + 1]
            end = shock_dates.index[i]
            shocks.append(
                {
                    "start": start,
                    "end": end,
                }
            )
            start = next_start

        # last shock
        shocks.append(
            {
                "start": start,
                "end": shock_dates.index[-1],
            }
        )
        return shocks

    @staticmethod
    def plot_shocks(df, shocks, pair, columns):
        fig, axs = plt.subplots(len(columns))
        fig.suptitle(pair)

        for idx, key in enumerate(columns):
            axs[idx].plot(df[key])
            axs[idx].set_title(key)
            axs[idx].grid(True)

            for i, shock in enumerate(shocks):
                axs[idx].axvline(shock["start"], alpha=0.5, color="red")
                axs[idx].axvline(shock["end"], alpha=0.5, color="red")
                label_position = 0.5 if i % 2 == 0 else -0.5  # alternate labels up/down
                # TODO: allow annotations
                # plt.text(shock["start"], label_position, shock["duration"])
        plt.show()

    def fit(
        self,
        df: pd.DataFrame,
        window=250,
        max_workers=8,
    ) -> pd.DataFrame:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            fitted = executor.map(
                BaseDataset.fit_levy,
                sliding_window_view(
                    np.array(df["log_returns"].values), window_shape=window
                ),
            )

        fitted = np.array(list(fitted))
        for i, col in enumerate(["alpha", "beta", "sigma", "mu_0", "mu_1"]):
            df[col] = np.concatenate([[np.nan] * (window - 1), fitted[:, i]])

        return df.dropna()

    @staticmethod
    def fit_levy(log_returns):
        init_fit = {"alpha": 2, "beta": 0, "sigma": 1, "mu": 0, "parameterization": 1}
        dist = pystable.create(
            init_fit["alpha"],
            init_fit["beta"],
            init_fit["sigma"],
            init_fit["mu"],
            init_fit["parameterization"],
        )
        pystable.fit(dist, log_returns, len(log_returns))
        return [
            dist.contents.alpha,
            dist.contents.beta,
            dist.contents.sigma,
            dist.contents.mu_0,
            dist.contents.mu_1,
        ]

    @staticmethod
    def plot_fit(df, shocks=None):
        columns = "returns close alpha beta".split()
        fig, axs = plt.subplots(len(columns))
        fig.suptitle("BTC/USD")
        df = df.dropna()
        for idx, key in enumerate(columns):
            axs[idx].plot(df[key])
            axs[idx].set_title(key)
            axs[idx].grid(True)

            if shocks is not None:
                for i, shock in enumerate(shocks):
                    axs[idx].axvline(shock["start"], alpha=0.5, color="red")
                    axs[idx].axvline(shock["end"], alpha=0.5, color="red")

        plt.show()

    def build_dataset(
        self,
        dir_path: str,
        start_date: str,
        end_date: str,
        freq: str = "5m",
        shocks_window: int = 1000,
        fit_window: int = 250,
        std_from_mean: float = 3.0,
        max_workers: int = 8,
        pre_shock_offset: int = 5,
        post_shock_offset: int = 5,
    ):
        # load, fit and find shocks
        date_to_int = lambda x: int(x.split(".")[0].replace("-", ""))
        start_date = int(start_date.replace("-", ""))
        end_date = int(end_date.replace("-", ""))
        # files = [
        #     dir_path + file
        #     for file in os.listdir(dir_path)
        #     if start_date <= date_to_int(file) < end_date
        # ]

        fitted_dfs = []
        shocks = []
        #for file in files:

        processed = self.preprocess(dir_path, freq)
        fitted = self.fit(
            df=processed,
            window=min(len(processed), fit_window),
            max_workers=max_workers,
        )
        fitted_dfs.append(fitted)
        for start in range(0, len(fitted) - shocks_window, shocks_window):
            filtered = BaseDataset.filter_data(
                fitted, fitted.index[start], fitted.index[start + shocks_window]
            )
            shocks.append(
                BaseDataset.find_shocks(filtered, std_from_mean=std_from_mean)
            )

        data = pd.concat(fitted_dfs, axis=0)
        shocks = np.concatenate(shocks)
        # maybe be useful in ts_fresh, don't need it now
        # data["is_shock"] = BaseDataset.tag_shocks(
        #     data, shocks, pre_shock_offset, post_shock_offset
        # )
        return data, shocks

    @staticmethod
    def tag_shocks(data, shocks, pre_shock_offset, post_shock_offset):
        # id = 1 for points near shocks, 0 otherwise
        data["is_shock"] = [0] * len(data)
        shocks_idxs = []
        for shock_idx in [data.index.get_loc(s["start"]) for s in shocks]:
            shocks_idxs.extend(
                [
                    data.index[i]
                    for i in range(
                        shock_idx - pre_shock_offset,
                        shock_idx + post_shock_offset,
                    )
                ]
            )

        return [1 if el in shocks_idxs else 0 for el in data.index]

    def old_build_dataset(
        self,
        shocks_window=1000,
        fit_window=250,
        std_from_mean=3.5,
        max_workers=8,
        from_checkpoint=False,
    ):

        checkpoint_path = from_root("data", "checkpoints")
        if from_checkpoint:
            with open(f"{checkpoint_path}/{self.pair}_{self.freq}.pkl", "rb") as f:
                file = pickle.load(f)
                self.fitted, self.shocks = file["data"], file["shocks"]
        else:
            # 1. detect shocks for all data
            for start in range(0, len(self.data) - shocks_window, shocks_window):
                sliced = self.data.iloc[start : start + shocks_window]
                self.find_shocks(
                    start_date=sliced.index[0],
                    end_date=sliced.index[-1],
                    std_from_mean=std_from_mean,
                )

            print("fitting data...")
            # 2. fit data
            self.fit(window=fit_window, max_workers=max_workers)
            with open(f"{checkpoint_path}/{self.pair}_{self.freq}.pkl", "wb") as f:
                pickle.dump(
                    {"data": self.fitted, "shocks": self.shocks},
                    f,
                )
            print("Saved checkpoint.")

        # 3. create features:
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

        # avg pct change in alpha, beta, price, volume  at 5, 10, 50 observations before shock
        cols = [
            "alpha",
            "beta",
            "sigma",
            "mu_0",
            "mu_1",
            "volume",
            "close",
        ]  # close should be last column
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
            "tot_pct_change": lambda x: 100 * (x[:, -1] - x[:, 0]) / x[:, 0]
            if x.shape[-1] > 0
            else None,
        }

        times = self.fitted.index.tolist()
        starting_time = self.fitted.index[0]
        shocks_indexes = []
        shock_features = []
        np_data = self.fitted[cols].to_numpy().T
        for shock in tqdm(self.shocks):
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
                if feature is not None:
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
            shock_feature["time"] = shock_index
            shock_features.append(shock_feature)
        print("\nFitted\n")
        # add non shocks samples
        non_shock_indexes = [
            i
            for i in range(observations_before_shocks[-1] + 5 + 1, len(times))
            if i not in shocks_indexes and times[i] > starting_time
        ]

        for non_shock_index in tqdm(
            random.sample(
                non_shock_indexes, min(len(self.fitted) // 2, 50 * len(self.shocks))
            )
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
