import pandas as pd
import os

# AWS_BUCK ET_PATH = "s3://wagon-public-datasets.s3.amazonaws.com/taxi-fare-test.csv"
abs_path = '/Users/melissasiddle/code/melissasiddle/TaxiFareModel/raw_data/train_1k.csv'

def get_data(nrows=1_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(abs_path, nrows=nrows)
    return df

def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data()