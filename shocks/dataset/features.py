import numpy as np
import random


class Features:
    def __init__(self, fitted_data, shocks):
        self.data = fitted_data
        self.columns = list(fitted_data.columns)
        self.shocks = shocks

    def filter_columns(self, condition: callable):
        return [self.columns.index(col) for col in self.columns if condition(col)]

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

    def vbuy_resiliency_5(self, x: np.array) -> np.array:
        """Order book resiliency at 5 ticks as defined in
        https://www.mds.deutsche-boerse.com/resource/blob/1334528/fdbd37665df6fa910df27172fc69c7ca/data/White-Paper-Risk-Alerts.pdf"""
        cols = self.filter_columns(lambda col: "vbuy" in col and int(col[-1]) <= 5)
        return np.sum(x[cols, :], axis=1)

    def vsell_resiliency_5(self, x: np.array) -> np.array:
        cols = self.filter_columns(lambda col: "vsell" in col and int(col[-1]) <= 5)
        return np.sum(x[cols, :], axis=1)

    def vbuy_resiliency_10(self, x: np.array) -> np.array:
        """Order book resiliency at 10 ticks"""
        cols = self.filter_columns(lambda col: "vbuy" in col)
        return np.sum(x[cols, :], axis=1)

    def vsell_resiliency_10(self, x: np.array) -> np.array:
        cols = self.filter_columns(lambda col: "vsell" in col)
        return np.sum(x[cols, :], axis=1)

    def sell_orderbook_resiliency_5(self, x: np.array) -> np.array:
        """Total cost needed to move price by 5 ticks. Computed as sum over i of volume_i * price_i"""
        vsell_cols = self.filter_columns(
            lambda col: "vsell" in col and int(col[-1]) <= 5
        )
        psell_cols = self.filter_columns(
            lambda col: "psell" in col and int(col[-1]) <= 5
        )
        return np.sum(x[vsell_cols, :] * x[psell_cols, :], axis=1)

    def buy_orderbook_resiliency_5(self, x: np.array) -> np.array:
        """Total cost needed to move price by 5 ticks. Computed as sum over i of volume_i * price_i"""
        vbuy_cols = self.filter_columns(lambda col: "vbuy" in col and int(col[-1]) <= 5)
        pbuy_cols = self.filter_columns(lambda col: "pbuy" in col and int(col[-1]) <= 5)
        return np.sum(x[vbuy_cols, :] * x[pbuy_cols, :], axis=1)

    def sell_orderbook_resiliency_10(self, x: np.array) -> np.array:
        """Total cost needed to move price by 5 ticks. Computed as sum over i of volume_i * price_i"""
        vsell_cols = self.filter_columns(lambda col: "vsell" in col)
        psell_cols = self.filter_columns(lambda col: "psell" in col)
        return np.sum(x[vsell_cols, :] * x[psell_cols, :], axis=1)

    def buy_orderbook_resiliency_10(self, x: np.array) -> np.array:
        """Total cost needed to move price by 5 ticks. Computed as sum over i of volume_i * price_i"""
        vbuy_cols = self.filter_columns(lambda col: "vbuy" in col)
        pbuy_cols = self.filter_columns(lambda col: "pbuy" in col)
        return np.sum(x[vbuy_cols, :] * x[pbuy_cols, :], axis=1)

    def quoted_spread(self, x: np.array) -> np.array:
        pbuy = self.columns.index("pbuy1")
        psell = self.columns.index("psell1")
        midpoint = (x[pbuy, :] + x[psell, :]) / 2
        return (x[psell, :] - x[pbuy, :]) / midpoint

    def compute_one_feature(
        self,
        data: np.array,
        feature: callable,
        shock_idx: int,
        offset: int,
    ):
        """compute the feature at offset steps before shock_idx. If columns is None all columns are selected"""

        sliced_data = data[:, shock_idx - offset - 1 : shock_idx]
        return feature(sliced_data)

    def create_name(self, feature: callable, cols: list | str, offset: int) -> list:
        """create name for feature. e.g. build_name(mean, ["alpha", "beta"], 5)
        returns ["alpha_5_mean", "beta_5_mean"]"""
        if not isinstance(cols, list):
            cols = [cols]
        return [f"{col}_{offset}_{feature.__name__}" for col in cols]

    def compute(
        self,
        pre_shock_offset: int = 5,
        post_shock_offset: int = 5,
        feature_offsets: list | tuple = (5, 10, 25, 50),
        non_shocks_ratio: int = 50,
    ) -> tuple:
        """Compute features for dataset

        :param cols: columns for which compute features
        :param pre_shock_offset: how many observations in advance we want to be notified of the shock event
        :param post_shock_offset: how many observations after shock has happened we want to skip before processing another shock.
                E.g. if a shock happened at time t, we will skip all shocks between t + 1 and t + post_shock_offset
        :param feature_offsets: how many observations before shock_offset we want to compute the features
        :param non_shocks_ratio: how many non_shocks event per shocks we want to have in the dataset. Defaults to 50.

        :return: featurized_shocks, featurized_non_shocks: dictionaries of features
        """

        np_data = self.data.to_numpy().T
        times = self.data.index.tolist()
        starting_time = self.data.index[0]
        valid_shocks = [s for s in self.shocks if s["start"] > starting_time]
        shock_indexes = [times.index(shock["start"]) for shock in valid_shocks]
        price_col = self.columns.index("price")
        # [(feature, names)]
        features_to_compute = self.features_names()

        non_shocks_indexes = self.get_non_shocks_indexes(
            non_shocks_ratio,
            valid_shocks,
            times,
            feature_offsets,
            pre_shock_offset,
            post_shock_offset,
        )

        featurized_shocks = self.compute_all_features(
            shock_indexes,
            times,
            np_data,
            price_col,
            feature_offsets,
            pre_shock_offset,
            features_to_compute,
        )

        featurized_non_shocks = self.compute_all_features(
            non_shocks_indexes,
            times,
            np_data,
            price_col,
            feature_offsets,
            pre_shock_offset,
            features_to_compute,
            is_shock=False,
        )

        return featurized_shocks, featurized_non_shocks

    def compute_all_features(
        self,
        indexes,
        times,
        np_data,
        price_col,
        feature_offsets,
        pre_shock_offset,
        features_to_compute,
        is_shock=True,
    ):

        events = []
        for idx in indexes:
            event_features = {
                "time": times[idx],
                "direction": 0
                if not is_shock
                else self.direction(np_data, price_col, idx),
            }
            for func, name in features_to_compute:
                for feature_offset in feature_offsets:
                    features = self.compute_one_feature(
                        np_data,
                        func,
                        idx - pre_shock_offset,
                        feature_offset,
                    )
                    names = self.create_name(
                        func,
                        name,
                        feature_offset,
                    )
                    # merge dictionaries
                    event_features = event_features | dict(zip(names, features))

            events.append(event_features)

        return events

    def get_non_shocks_indexes(
        self,
        non_shocks_ratio,
        valid_shocks,
        times,
        feature_offsets,
        pre_shock_offset,
        post_shock_offset,
    ):
        shocks_indexes = [
            i
            for shock in valid_shocks
            for i in range(
                times.index(shock["start"]) - pre_shock_offset,
                times.index(shock["start"]) + post_shock_offset,
            )
        ]
        # the range is chosen so that we don't include nans
        non_shock_indexes = [
            s
            for s in range(feature_offsets[-1] + post_shock_offset + 1, len(times))
            if s not in shocks_indexes
        ]
        size = min(len(non_shock_indexes), non_shocks_ratio * len(self.shocks))
        return random.sample(non_shock_indexes, size)

    def features_names(self):
        cols_no_direction = [col for col in self.columns if col != "direction"]
        return [
            (self.mean_pct_change, cols_no_direction),
            (self.std_pct_change, cols_no_direction),
            (self.mean, cols_no_direction),
            (self.std, cols_no_direction),
            (
                self.buy_orderbook_resiliency_5,
                "",
            ),
            (
                self.sell_orderbook_resiliency_5,
                "",
            ),
            (
                self.buy_orderbook_resiliency_10,
                "",
            ),
            (
                self.sell_orderbook_resiliency_10,
                "",
            ),
            (
                self.vbuy_resiliency_5,
                "",
            ),
            (self.vsell_resiliency_5, ""),
            (
                self.vbuy_resiliency_10,
                "",
            ),
            (self.vsell_resiliency_10, ""),
        ]
