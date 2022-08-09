from enum import Enum
from datetime import datetime, timedelta
import pandas as pd 
import os 
import re 
import gc

import warnings
warnings.filterwarnings("ignore")

class Granularity(Enum):
    """ The possible Granularity to build the OHLC old_data from lob """
    Sec1 = "1S"
    Sec5 = "5S"
    Sec15 = "15S"
    Sec30 = "30S"
    Min1 = "1Min"
    Min5 = "5Min"
    Min15 = "15Min"
    Min30 = "30Min"
    Hour1 = "1H"
    Hour2 = "2H"
    Hour6 = "6H"
    Hour12 = "12H"
    Day1 = "1D"
    Day2 = "2D"
    Day5 = "7D"
    Month1 = "30D"

class OrderEvent(Enum):
    """event types of orderbook"""
    submission = 1
    cancellation = 2
    deletion = 3
    execution_visible = 4
    execution_hidden = 5
    cross_trade = 6
    halt = 7


def orderbook_columns(level: int):
    """ return the column names for the LOBSTER orderbook, acording the input level """
    orderbook_columns = []
    for i in range(1, level + 1):
        orderbook_columns += ["psell" + str(i), "vsell" + str(i), "pbuy" + str(i), "vbuy" + str(i)]
    return orderbook_columns

def message_columns():
    """ return the message columns for the LOBSTER orderbook """
    return ["time", "event_type", "order_id", "size", "price", "direction", "unk"]



def lobster_to_sec_df(message_df, orderbook_df,
                      datetime_start: str,
                      granularity: Granularity = Granularity.Sec1,
                      level: int = 10, 
                      add_messages=True):
    """ create a dataframe with midprices, sell and buy for each second

        message_df : a csv df with the messages (lobster old_data format) without initial start lob
        ordebook_df : a csv df with the orderbook (lobster old_data format) without initial start lob
        datetime_start : should be a start date in the message file and orderbook file
        granularity : the granularity to use in the mid-prices computation
        plot : whether print or not the mid_prices
        level : the level of the old_data
        add_messages : if True keep messages along the orderbook data. It does not work with granularity != None
    """
    start_date = datetime.strptime(datetime_start, "%Y-%m-%d")

    # to be sure that columns are okay
    orderbook_df.columns = orderbook_columns(level)
    message_df.columns = message_columns()

    # convert the time to seconds and structure the df to the input granularity
    orderbook_df["seconds"] = message_df["time"]

    if add_messages and granularity is not None:
        orderbook_df[message_df.columns] = message_df[message_df.columns]
        accepted_orders = [o.value for o in (OrderEvent.execution_visible, OrderEvent.submission, OrderEvent.execution_hidden)]
        orderbook_df = orderbook_df[orderbook_df["event_type"].isin(accepted_orders)]

    orderbook_df["date"] = [start_date + timedelta(seconds=i) for i in orderbook_df["seconds"]]

    if granularity is not None:
        orderbook_df.set_index("date", inplace=True)
        # volume_cols = [col for orderbook_df.columns if "vsell" in col or "vbuy" in col]
        orderbook_df = orderbook_df.resample(granularity.value).first()
        orderbook_df.reset_index(inplace=True)

    orderbook_df = orderbook_df.sort_values(by="date").reset_index(drop=True).copy()
    orderbook_df.drop(columns=['seconds'], inplace=True)

    return orderbook_df.set_index('date')




def read_sub_routine(file_7z: str, first_date: str = "1990-01-01",
                     last_date: str = "2100-01-01",
                     type_file: str = "orderbook",
                     level: int = 10,
                     path: str = "", 
                     date: str= "") -> dict:
    """
        :param file_7z: the input file where the csv with old_data are stored
        :param first_date: the first day to load from the input file
        :param last_date: the last day to load from the input file
        :param type_file: the kind of old_data to read. type_file in ("orderbook", "message")
        :param level: the LOBSTER level of the orderbook
        :param path: data path
        :return: a dictionary with {day : dataframe}
    """
    assert type_file in ("orderbook", "message"), "The input type_file: {} is not valid".format(type_file)

    columns = message_columns() if type_file == "message" else orderbook_columns(level)
    # if both none then we automatically detect the dates from the files
    first_date = datetime.strptime(first_date, "%Y-%m-%d")
    last_date = datetime.strptime(last_date, "%Y-%m-%d")

    all_period = {}  # day :  df

    path = path + file_7z

    ## mine 
    file = [ el for el in sorted(os.listdir(path)) if date in el and type_file in el][0]
    curr = path + '/' + file
    df = pd.read_csv(curr, names=columns)
    return df
    ## mine

    for file in [ el for el in sorted(os.listdir(path)) if date in el] :
        # read only the selected type of file
        if type_file not in str(file):
            continue

        # read only the old_data between first_ and last_ input dates
        m = re.search(r".*([0-9]{4}-[0-9]{2}-[0-9]{2}).*", str(file))
        if m:
            entry_date = datetime.strptime(m.group(1), "%Y-%m-%d")
            if entry_date < first_date or entry_date > last_date:
                continue
        else:
            print("error for file: {}".format(file))
            continue

        curr = path + '/' + file
        df = pd.read_csv(curr, names=columns)
        # put types
        all_period[entry_date] = df

    return all_period



def from_folder_to_unique_df(file_7z: str, 
                             first_date: str = "1990-01-01",
                             last_date: str = "2100-01-01",
                             plot: bool = False, 
                             level: int = 10,
                             path: str = "",
                             granularity: Granularity = Granularity.Sec1,
                             add_messages = True, 
                             name=None):
    """ return a unique df with also the label

        add_messages : if True keep messages along the orderbook data. It does not work with granularity != None

    """
    # message_dfs = read_sub_routine(file_7z, first_date, last_date, "message", level=level, path=path)
    # orderbook_dfs = read_sub_routine(file_7z, first_date, last_date, "orderbook", level=level, path=path)
 
    # frames = []
 
    # assert list(message_dfs.keys()) == list(orderbook_dfs.keys()), "the messages and orderbooks have different days!!"
    files = [el for el in sorted(os.listdir(file_7z)) if "2021" not in el and "2022-01" not in el]
    # for d in message_dfs.keys():

    save_path = f"/home/mercanti/shocks/data/{name}_aggregated/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    for file in files:
        date = file.split("_")[1]
        message_dfs = read_sub_routine(file_7z, date, date, "message", level=level, path=path, date=date)
        orderbook_dfs = read_sub_routine(file_7z, date, date, "orderbook", level=level, path=path, date=date)
        print("processing", date, "records: ", len(message_dfs), len(orderbook_dfs))
        try: 
            tmp_df = lobster_to_sec_df(
                message_dfs, orderbook_dfs, date, granularity=granularity,
                level=level, add_messages=add_messages)
            # frames.append(tmp_df)
            print("processed",  date)
            if len(tmp_df) > 0:
                print("saving",  date)

                tmp_df.to_csv(f"{save_path}{date}.csv")
                del tmp_df
                gc.collect()
        except TypeError:
            print(f"day {date} is empty")


    # result = pd.concat(frames, ignore_index=False)

   # return result

if __name__ == "__main__":

    print("Processing...")
    name = "GME"
    df = from_folder_to_unique_df(f"/home/mercanti/shocks/data/lobster/{name}/", first_date="2022-01-27", level=10, name=name)
    print("working on apple")
    name = "AAPL"
    df = from_folder_to_unique_df(f"/home/mercanti/shocks/data/lobster/{name}/", first_date="2022-01-27", level=10, name=name)

    print("Done.")
