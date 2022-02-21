import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin

def lat_lon_distance(df:pd.DataFrame) -> pd.DataFrame:
    """Calculate the flight distance from the latitude and longitude data

    Args:
        df (pd.DataFrame): dataframe containing the zindi data and the merged airports data

    Returns:
        pd.DataFrame: df with distance column
    """
    point1 = (df.lat_DEP.values, df.lon_DEP.values)
    point2 = (df.lat_ARR.values, df.lon_ARR.values)

    dist = []
    for num in range(len(point1[0])):
        pt1 = point1[0][num], point1[1][num]
        pt2 = point2[0][num], point2[1][num]
        dist.append(geodesic(pt1, pt2).km)
    df["distance"] = dist
    return df

def feature_engineering(df:pd.DataFrame, df_test:pd.DataFrame) -> pd.DataFrame:
    """Create custom features in the dataset

    Args:
        df (pd.DataFrame): Input full dataset from zindi
        df_test (pd.DataFrame): Test dataset to perform the same operations on

    Returns:
        pd.DataFrame: dataframe with added features
    """
    # Calculate distances
    df = lat_lon_distance(df)
    df_test = lat_lon_distance(df_test)
    # Initialize custom transformer
    features_enable = [1, 1, 1, 1, 1] # [domestic, dep_hour, dep_weekday, duration_min, operator]
    # Pipeline to add features
    attr_addr = flight_preprocessor()
    df = attr_addr.fit_transform(df)
    df_test = attr_addr.transform(df_test)
    return df, df_test

def create_feature(df:pd.DataFrame, features_enable=[1, 1, 1, 1, 1, 1, 1, 1]) -> pd.DataFrame:
    """Add custom features to the dataset

    Args:
        df (pd.DataFrame): dataframe containing the zindi data and the merged airports data
        features (list, optional): List specifying which features to create.
        Defaults to [1, 1, 1, 1, 1].

    Returns:
        pd.DataFrame: final dataframe
    """
    if features_enable[0]:
        df["domestic"] = (df.country_DEP == df.country_ARR).astype("int")
    if features_enable[1]:
        df["dep_hour"] = df["STD"].dt.hour
    if features_enable[2]:
        df["dep_weekday"] = df.STD.dt.weekday
    if features_enable[3]:
        df["duration_min"] = (df.STA - df.STD).dt.total_seconds() / 60
    if features_enable[4]:
        df["operator"] = df.FLTID.str.replace('\d+', '')
    if features_enable[5]:
        df["arr_hour"] = df["STA"].dt.hour
    if features_enable[6]:
        df["dep_day"] = df["STD"].dt.dayofyear + 365 * (df["STD"].dt.year - 2016)
    if features_enable[7]:
        df["year"] = df["STD"].dt.year
    return df

def remove_outliers(df:pd.DataFrame, feat, sigma=2) -> pd.DataFrame:
    """Remove outliers from the specified column

    Args:
        df (pd.DataFrame): dataframe containing the zindi data and the merged airports data
        feat (string): name of the column to process
        sigma (int, optional): Range of distribution to remove. Defaults to 2.

    Returns:
        pd.DataFrame: dataframe with outliers removed
    """
    df = df[df[feat] < (df[feat].median() + sigma * df[feat].std())]
    return df

# def flight_preprocessor(df:pd.DataFrame, features_enable=[1, 1, 1, 1, 1]) -> pd.DataFrame:
#     df = create_feature(df, features_enable)
#     return df

class flight_preprocessor(BaseEstimator, TransformerMixin):
    """preprocessor to be used with sklearn pipelines in order to
    check relevance of features using GridSearchCV
    """
    def __init__(self, features_enable=[1, 1, 1, 1, 1, 1, 1, 1]):
        self.features_enable = features_enable

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = create_feature(X, self.features_enable)
        #   cols_to_drop = ["STA", "STD"]
        #   X = drop_column(X, cols_to_drop)
        return X
