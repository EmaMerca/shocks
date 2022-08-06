from shocks.dataset import *
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import json

if __name__ == "__main__":

    symbol = "aapl"
    start_date="2021-11-02"
    end_date = "2021-11-03"
    freq = "30s"
    dataset = LobsterDataset("aapl")
    data, shocks = dataset.build_dataset(
        dir_path="/home/ema/dev/shocks/data/lobster/AAPL/",
        start_date=start_date,
        end_date=end_date,
        freq=freq,
        shocks_window=300,
        fit_window=300,
        std_from_mean=2.5,
    )
    features = Features(data, shocks)
    f = features.compute(
        pre_shock_offset=5, post_shock_offset=5, feature_offsets=[5, 10]
    )

    with open(f"features_{symbol}_{start_date}_{end_date}_{freq}.json", "w") as file:
        json.dump(f, file)
