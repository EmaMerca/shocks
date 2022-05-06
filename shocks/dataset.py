import os

import pandas as pd
import datetime
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pystable


from shocks import *

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
            for fname in [os.path.join(pair_dir, file) for file in os.listdir(pair_dir)]
        ]
        return pd.concat(df_list, axis=0)

    @staticmethod
    def resample_data(
        df: pd.DataFrame, freq: str = "1h", mode: str = "last"
    ) -> pd.DataFrame:
        ohlc_dict = {
            "close": mode,
            "volume": "sum"
        }
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
        assert self.data.columns.tolist() == list(keep_cols), "columns are missing, consider adding them in the ohlc_dict in the resample method"
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

        def tot_pct_change(klass):
            # using klass instead of self not to create confusion with Dataset class' self
            to_list = klass.tolist()
            return (to_list.__getitem__(-1) - to_list.__getitem__(0)) / to_list.__getitem__(0)

        def compute_feature(df, shock_idx, idx_before_shock, col, *args):
            """*args are df class methods which can be used with getattr()"""
            df_slice = df.iloc[shock_idx - idx_before_shock - 1: shock_idx][col]
            for arg in args:
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
        #           avg pct change in alpha, beta  at 5, 10, 50 observations before shock
        #           avg pct change in price, volume  at 5, 10, 50 observations before shock


        # self.fitted is a pandas dataframe
        times = self.fitted.index.tolist()
        starting_time = self.fitted.index[0]
        shock_features = []
        for shock in self.shocks:
            if shock["start"] <= starting_time:
                continue
            shock_index = (
                times.index(shock["start"]) - 5
            )  # shock signal should fire off 5 observations before shock happens else it's too late


            shock_features.append(
                {
            "time": shock["start"],
            #
            "alpha_5_mean": self.fitted.iloc[shock_index - 5 - 1 : shock_index]["alpha"].mean(),
            "alpha_5_std": self.fitted.iloc[shock_index - 5 - 1 : shock_index]["alpha"].std(),
            "alpha_10_mean": self.fitted.iloc[shock_index - 10 - 1 : shock_index]["alpha"].mean(),
            "alpha_10_std": self.fitted.iloc[shock_index - 10 - 1 : shock_index]["alpha"].std(),
            "alpha_25_mean": self.fitted.iloc[shock_index - 25 - 1 : shock_index]["alpha"].mean(),
            "alpha_25_std": self.fitted.iloc[shock_index - 25 - 1 : shock_index]["alpha"].std(),
            "alpha_50_mean": self.fitted.iloc[shock_index - 50 - 1 : shock_index]["alpha"].mean(),
            "alpha_50_std": self.fitted.iloc[shock_index - 50 - 1 : shock_index]["alpha"].std(),
            #
            "alpha_pct_change_5_mean": self.fitted.iloc[shock_index - 5 - 1 : shock_index]["alpha"].pct_change.mean(),
            "alpha_pct_change_5_std": self.fitted.iloc[shock_index - 5 - 1 : shock_index]["alpha"].std(),
            "alpha_pct_change_10_mean": self.fitted.iloc[shock_index - 10 - 1 : shock_index]["alpha"].mean(),
            "alpha_pct_change_10_std": self.fitted.iloc[shock_index - 10 - 1 : shock_index]["alpha"].std(),
            "alpha_pct_change_25_mean": self.fitted.iloc[shock_index - 25 - 1 : shock_index]["alpha"].mean(),
            "alpha_pct_change_25_std": self.fitted.iloc[shock_index - 25 - 1 : shock_index]["alpha"].std(),
            "alpha_pct_change_50_mean": self.fitted.iloc[shock_index - 50 - 1 : shock_index]["alpha"].mean(),
            "alpha_pct_change_50_std": self.fitted.iloc[shock_index - 50 - 1 : shock_index]["alpha"].std(),
            # 
            "alpha_tot_change_5":  total_pct_change(self.fitted.iloc[shock_index - 5 - 1 : shock_index]["alpha"].dropna().tolist()),
            "alpha_tot_change_10": total_pct_change(self.fitted.iloc[shock_index - 10 - 1 : shock_index]["alpha"].dropna().tolist()),
            "alpha_tot_change_25": total_pct_change(self.fitted.iloc[shock_index - 25 - 1 : shock_index]["alpha"].dropna().tolist()),
            "alpha_tot_change_50": total_pct_change(self.fitted.iloc[shock_index - 50 - 1 : shock_index]["alpha"].dropna().tolist()),
            #
            #
            "beta_5_mean": self.fitted.iloc[shock_index - 5 - 1: shock_index]["beta"].mean(),
            "beta_5_std": self.fitted.iloc[shock_index - 5 - 1: shock_index]["beta"].std(),
            "beta_10_mean": self.fitted.iloc[shock_index - 10 - 1: shock_index]["beta"].mean(),
            "beta_10_std": self.fitted.iloc[shock_index - 10 - 1: shock_index]["beta"].std(),
            "beta_25_mean": self.fitted.iloc[shock_index - 25 - 1: shock_index]["beta"].mean(),
            "beta_25_std": self.fitted.iloc[shock_index - 25 - 1: shock_index]["beta"].std(),
            "beta_50_mean": self.fitted.iloc[shock_index - 50 - 1: shock_index]["beta"].mean(),
            "beta_50_std": self.fitted.iloc[shock_index - 50 - 1: shock_index]["beta"].std(),
            #
            "beta_pct_change_5_mean": self.fitted.iloc[shock_index - 5 - 1: shock_index]["beta"].mean(),
            "beta_pct_change_5_std": self.fitted.iloc[shock_index - 5 - 1: shock_index]["beta"].std(),
            "beta_pct_change_10_mean": self.fitted.iloc[shock_index - 10 - 1: shock_index]["beta"].mean(),
            "beta_pct_change_10_std": self.fitted.iloc[shock_index - 10 - 1: shock_index]["beta"].std(),
            "beta_pct_change_25_mean": self.fitted.iloc[shock_index - 25 - 1: shock_index]["beta"].mean(),
            "beta_pct_change_25_std": self.fitted.iloc[shock_index - 25 - 1: shock_index]["beta"].std(),
            "beta_pct_change_50_mean": self.fitted.iloc[shock_index - 50 - 1: shock_index]["beta"].mean(),
            "beta_pct_change_50_std": self.fitted.iloc[shock_index - 50 - 1: shock_index]["beta"].std(),
            #
            "beta_tot_change_5": total_pct_change(self.fitted.iloc[shock_index - 5 - 1: shock_index]["beta"].dropna().tolist()),
            "beta_tot_change_10": total_pct_change(self.fitted.iloc[shock_index - 10 - 1: shock_index]["beta"].dropna().tolist()),
            "beta_tot_change_25": total_pct_change(self.fitted.iloc[shock_index - 25 - 1: shock_index]["beta"].dropna().tolist()),
            "beta_tot_change_50": total_pct_change(self.fitted.iloc[shock_index - 50 - 1: shock_index]["beta"].dropna().tolist()),
            #
            #
            "price_5_mean": self.fitted.iloc[shock_index - 5 - 1: shock_index]["close"].mean(),
            "price_5_std": self.fitted.iloc[shock_index - 5 - 1: shock_index]["close"].std(),
            "price_10_mean": self.fitted.iloc[shock_index - 10 - 1: shock_index]["close"].mean(),
            "price_10_std": self.fitted.iloc[shock_index - 10 - 1: shock_index]["close"].std(),
            "price_25_mean": self.fitted.iloc[shock_index - 25 - 1: shock_index]["close"].mean(),
            "price_25_std": self.fitted.iloc[shock_index - 25 - 1: shock_index]["close"].std(),
            "price_50_mean": self.fitted.iloc[shock_index - 50 - 1: shock_index]["close"].mean(),
            "price_50_std": self.fitted.iloc[shock_index - 50 - 1: shock_index]["close"].std(),
            #
            "price_pct_change_5_mean": self.fitted.iloc[shock_index - 5 - 1: shock_index]["close"].mean(),
            "price_pct_change_5_std": self.fitted.iloc[shock_index - 5 - 1: shock_index]["close"].std(),
            "price_pct_change_10_mean": self.fitted.iloc[shock_index - 10 - 1: shock_index]["close"].mean(),
            "price_pct_change_10_std": self.fitted.iloc[shock_index - 10 - 1: shock_index]["close"].std(),
            "price_pct_change_25_mean": self.fitted.iloc[shock_index - 25 - 1: shock_index]["close"].mean(),
            "price_pct_change_25_std": self.fitted.iloc[shock_index - 25 - 1: shock_index]["close"].std(),
            "price_pct_change_50_mean": self.fitted.iloc[shock_index - 50 - 1: shock_index]["close"].mean(),
            "price_pct_change_50_std": self.fitted.iloc[shock_index - 50 - 1: shock_index]["close"].std(),
            #
            "price_tot_change_5": total_pct_change(self.fitted.iloc[shock_index - 5 - 1: shock_index]["close"].dropna().tolist()),
            "price_tot_change_10": total_pct_change(self.fitted.iloc[shock_index - 10 - 1: shock_index]["close"].dropna().tolist()),
            "price_tot_change_25": total_pct_change(self.fitted.iloc[shock_index - 25 - 1: shock_index]["close"].dropna().tolist()),
            "price_tot_change_50": total_pct_change(self.fitted.iloc[shock_index - 50 - 1: shock_index]["close"].dropna().tolist()),
            #
            #
            "volume_5_mean": self.fitted.iloc[shock_index - 5 - 1: shock_index]["volume"].mean(),
            "volume_5_std": self.fitted.iloc[shock_index - 5 - 1: shock_index]["volume"].std(),
            "volume_10_mean": self.fitted.iloc[shock_index - 10 - 1: shock_index]["volume"].mean(),
            "volume_10_std": self.fitted.iloc[shock_index - 10 - 1: shock_index]["volume"].std(),
            "volume_25_mean": self.fitted.iloc[shock_index - 25 - 1: shock_index]["volume"].mean(),
            "volume_25_std": self.fitted.iloc[shock_index - 25 - 1: shock_index]["volume"].std(),
            "volume_50_mean": self.fitted.iloc[shock_index - 50 - 1: shock_index]["volume"].mean(),
            "volume_50_std": self.fitted.iloc[shock_index - 50 - 1: shock_index]["volume"].std(),
            #
            "volume_pct_change_5_mean": self.fitted.iloc[shock_index - 5 - 1: shock_index]["volume"].mean(),
            "volume_pct_change_5_std": self.fitted.iloc[shock_index - 5 - 1: shock_index]["volume"].std(),
            "volume_pct_change_10_mean": self.fitted.iloc[shock_index - 10 - 1: shock_index]["volume"].mean(),
            "volume_pct_change_10_std": self.fitted.iloc[shock_index - 10 - 1: shock_index]["volume"].std(),
            "volume_pct_change_25_mean": self.fitted.iloc[shock_index - 25 - 1: shock_index]["volume"].mean(),
            "volume_pct_change_25_std": self.fitted.iloc[shock_index - 25 - 1: shock_index]["volume"].std(),
            "volume_pct_change_50_mean": self.fitted.iloc[shock_index - 50 - 1: shock_index]["volume"].mean(),
            "volume_pct_change_50_std": self.fitted.iloc[shock_index - 50 - 1: shock_index]["volume"].std(),
            #
            "volume_tot_change_5": total_pct_change(self.fitted.iloc[shock_index - 5 - 1: shock_index]["volume"].dropna().tolist()),
            "volume_tot_change_10": total_pct_change(self.fitted.iloc[shock_index - 10 - 1: shock_index]["volume"].dropna().tolist()),
            "volume_tot_change_25": total_pct_change(self.fitted.iloc[shock_index - 25 - 1: shock_index]["volume"].dropna().tolist()),
            "volume_tot_change_50": total_pct_change(self.fitted.iloc[shock_index - 50 - 1: shock_index]["volume"].dropna().tolist()),
                }
            )

        import pickle
        with open("./test.pkl", "wb") as f:
            pickle.dump({"features": shock_features, "data": self.fitted}, f)
        print("done")





















