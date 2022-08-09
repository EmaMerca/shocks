from shocks.dataset import *
import json

if __name__ == "__main__":
    symbol = "KO"
    start_date = "2021-11-01"
    end_date = "2022-01-01"
    freq = "30s"
    std_from_mean = 2

    dataset = LobsterDataset(symbol)
    data, shocks = dataset.build_dataset(
        dir_path=f"/home/ema/dev/shocks/data/lobster/{symbol}/",
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        shocks_window=300,
        fit_window=300,
        std_from_mean=std_from_mean,
    )
    features = Features(data, shocks)
    f = features.compute(
        pre_shock_offset=5, post_shock_offset=5, feature_offsets=[5, 10, 50, 100]
    )
    with open(
        f"/home/ema/dev/shocks/data/featurized/features__{symbol}__{start_date}_{end_date}__{freq}__{std_from_mean}_std.json",
        "w",
    ) as file:
        json.dump(f, file)
