import pandas as pd
import numpy as np

def fix_airport(df:pd.DataFrame) -> pd.DataFrame:
    """Berlin Schönefeld is referenced as SXF in the zindi data, but only 
    exists as BER in the airports data. Therefore we replace SXF with BER

    Args:
        df (pd.DataFrame): dataframe containing the zindi data

    Returns:
        pd.DataFrame: dataframe with replaced airports
    """
    df["DEPSTN"] = df["DEPSTN"].str.replace("SXF", "BER")
    df["ARRSTN"] = df["ARRSTN"].str.replace("SXF", "BER")
    return df

def merge_airports(df:pd.DataFrame, df_air:pd.DataFrame) -> pd.DataFrame:
    """Merge airportsdata to the zindi data

    Args:
        df (pd.DataFrame): dataframe containing the zindi data
        df_air (pd.DataFrame): dataframe containing the airportsdata
    Returns:
        pd.DataFrame: merged dataframe
    """
    df = df.merge(df_air, left_on="DEPSTN", right_on="iata", suffixes=("_DEP", "_ARR"))
    df = df.merge(df_air, left_on="ARRSTN", right_on="iata", suffixes=("_DEP", "_ARR"))
    df.sort_values("STD", inplace=True) # Sort values by departure time
    df.reset_index(drop=True, inplace=True) # Reset index
    return df

def drop_column(df:pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """Drop specified columns from dataframe

    Args:
        df (pd.DataFrame): dataframe containing the zindi data
        cols_to_drop (list): list of strings identifying the columns to drop

    Returns:
        pd.DataFrame: dataframe with dropped columns
    """
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

def load_prepare_flight_data():
    """Load the data and preprocess it

    Returns:
        pd.DataFrame: dataframe containing the cleaned data
    """
    df = pd.read_csv("data/Train.csv", parse_dates=[1,5,6])
    df_test = pd.read_csv("data/Test.csv", parse_dates=[1,5,6])
    df_air = pd.read_csv("data/airports.csv")
    # Fix wrong datetime parsing
    df["STA"] = pd.to_datetime(df["STA"], format="%Y-%m-%d %H.%M.%S")
    df_test["STA"] = pd.to_datetime(df_test["STA"], format="%Y-%m-%d %H.%M.%S")
    # Fix airport issue SXF/BER
    df = fix_airport(df)
    df_test = fix_airport(df_test)
    # Merge airportsdata
    df = merge_airports(df, df_air)
    df_test = merge_airports(df_test, df_air)
    return df, df_test