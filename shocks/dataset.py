import os

import pandas as pd
import datetime
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from shocks import *

FREQS = {
    "1m": "1T",
    "5m": "5T",
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
        }
        return df.resample(FREQS[freq], closed="left", label="left").apply(ohlc_dict).dropna()

    @staticmethod
    def filter_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        mask = (df.index > start_date) & (df.index < end_date)
        return df.loc[mask]

    @staticmethod
    def datetime_index(df: pd.DataFrame, time_col: str = "open time", is_millisec: bool=True) -> pd.DataFrame:
        def to_human_time(unix_time):
            if is_millisec:
                unix_time /= 1000
            t = datetime.datetime.utcfromtimestamp(unix_time)
            return f"{t.year}-{t.month}-{t.day} {t.hour}:{t.minute}"

        df.index = pd.to_datetime(df[time_col].apply(to_human_time))
        return df

    def preprocess(self, freq="1h", keep_cols: List[str] = ("close")) -> None:
        self.data = Dataset.resample_data(
            df=Dataset.datetime_index(df=self.data), freq=freq
        )
        self.data["close"] = self.data["close"].ffill()
        self.data = self.data[[keep_cols]]
        self.data["returns"] = self.data["close"].pct_change().dropna()

    def find_shocks(
        self,
        start_date: str,
        end_date: str,
        std_from_mean: int = 3,
        min_duration: int = 3,
        max_duration: int = 15,
        plot: bool = False,
    ):
        df = Dataset.filter_data(self.data, start_date, end_date)
        # find dates where returns distribution is more than std_from_mean away from mean
        upper_threshold = df["returns"].mean() + std_from_mean * df["returns"].std()
        lower_threshold = df["returns"].mean() - std_from_mean * df["returns"].std()
        shock_dates = df["returns"][(df["returns"] <= lower_threshold) | (df["returns"] >= upper_threshold)]

        # shocks is just a list of dates, we need to extract the individual shocks from there
        shocks = []
        start = shock_dates.index[0]
        for i in range(len(shock_dates.index) - 1):
            next_start = shock_dates.index[i + 1]
            end = shock_dates.index[i]
            if len(df.loc[start:next_start]) >= max_duration:
                shocks.append({"start": start, "end": end, "duration": end - start})
                start = next_start
        # last shock
        shocks.append(
            {
                "start": start,
                "end": shock_dates.index[-1],
                "duration": shock_dates.index[-1] - start,
            }
        )

        self.shocks = [
            el for el in shocks if len(df.loc[el["start"] : el["end"]]) >= min_duration
        ]

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
                    #plt.text(shock["start"], label_position, shock["duration"])
