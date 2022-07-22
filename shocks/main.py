from shocks.dataset import *

if __name__ == "__main__":
    dataset = LobsterDataset("aapl")
    data, shocks = dataset.build_dataset(
        dir_path="/home/ema/dev/shocks/data/lobster/AAPL/",
        start_date="2021-11-02",
        end_date="2021-11-03",
        freq="30s",
        shocks_window=300,
        fit_window=300,
        std_from_mean=2.5
    )
    len(shocks)
    features = Features(data, shocks)
    f = features.compute(pre_shock_offset=5, post_shock_offset=5, feature_offsets=[5, 10])
    len(f)
