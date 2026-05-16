from config import Train_Data, Processed_Train_Data

import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(Train_Data)
    return df

def data_cleaning(df):
    df = df.copy()
    df.drop(["Total day charge","Total eve charge","Total night charge","Total intl charge"],axis=1,inplace=True)
    return df

def save_processed_data(df):
    Processed_Train_Data.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Processed_Train_Data, index=False)

def main():
    df = load_data()
    df = data_cleaning(df)
    save_processed_data(df)
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    main()