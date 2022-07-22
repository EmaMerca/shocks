import numpy as np
import pandas as pd
from typing import List


class Features:
    def __init__(self, fitted_data, shocks):
        self.data = fitted_data
        self.shocks = shocks
        self.features_to_compute = [self.mean_pct_change, self.std_pct_change]

    def direction(self, x: np.array, price_col: int, shock_idx: int) -> int:
        """direction of the shock. 1 if shock causes price to increase, -1 otherwise"""
        return -1 if x[price_col, shock_idx - 1] >= x[price_col, shock_idx] else 1

    def mean(self, x: np.array) -> np.array:
        # x is a matrix, mean is computed columns-wise.
        # Replace inf with nan and ignore nans in mean
        ma = np.ma.masked_array(x, ~np.isfinite(x)).filled(np.nan)
        return np.nanmean(ma, axis=1)

    def std(self, x: np.array) -> np.array:
        # x is a matrix, std is computed columns-wise. inf or nan values are ignored
        ma = np.ma.masked_array(x, ~np.isfinite(x)).filled(np.nan)
        return np.nanstd(ma, axis=1)

    def pct_change(self, x: np.array) -> np.array:
        return np.diff(x) / x[:, :-1] * 100

    def tot_pct_change(self, x: np.array) -> np.array:
        return 100 * (x[:, -1] - x[:, 0]) / x[:, 0] if x.shape[-1] > 0 else None

    def mean_pct_change(self, x: np.array) -> np.array:
        return self.mean(self.pct_change(x))

    def std_pct_change(self, x: np.array) -> np.array:
        return self.std(self.pct_change(x))

    def compute_one(
        self, data: np.array, feature: callable, shock_idx: int, offset: int
    ):
        """compute the feature at offset steps before shock_idx"""
        sliced_data = data[:, shock_idx - offset - 1 : shock_idx]
        return feature(sliced_data)

    def create_name(self, feature: callable, cols: list, offset: int) -> list:
        """create name for feature. e.g. build_name(mean, ["alpha", "beta"], 5)
        returns ["alpha_5_mean", "beta_5_mean"]"""
        return [f"{col}_{offset}_{feature.__name__}" for col in cols]

    def compute(
        self,
        pre_shock_offset: int,
        post_shock_offset: int,
        feature_offsets: List[int],
        cols: List[str] = None,
    ):
        """Compute features for dataset

        :param cols: columns for which compute features
        :param pre_shock_offset: how many observations in advance we want to be notified of the shock event
        :param post_shock_offset: how many observations after shock has happened we want to skip before processing another shock.
                E.g. if a shock happened at time t, we will skip all shocks between t + 1 and t + post_shock_offset
        :param feature_offsets: how many observations before shock_offset we want to compute the features
        :return:
        """
        featurized_shocks = []
        if not cols:
            cols = list(self.data.columns)

        np_data = self.data[cols].to_numpy().T
        price_col = list(self.data.columns).index("price")
        times = self.data.index.tolist()
        starting_time = self.data.index[0]
        valid_shocks = [s for s in self.shocks if s["start"] > starting_time]

        for shock in valid_shocks:
            shock_idx = times.index(shock["start"])
            shock_features = {
                "time": shock["start"],
                "direction": self.direction(np_data, price_col, shock_idx),
            }
            for func in self.features_to_compute:
                for feature_offset in feature_offsets:
                    features = self.compute_one(
                        np_data,
                        func,
                        shock_idx - pre_shock_offset,
                        feature_offset,
                    )
                    names = self.create_name(func, cols, feature_offset)
                    shock_features = shock_features | dict(zip(names, features))

            featurized_shocks.append(shock_features)

        return featurized_shocks
