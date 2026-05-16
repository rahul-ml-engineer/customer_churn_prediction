import pandas as pd

from config import Processed_Train_Data, Feature_eng_Data

def load_data():
    df = pd.read_csv(Processed_Train_Data)
    return df

def feature_engineering(df):
    df = df.copy()

    df["Total_Calls"] = df["Total day calls"]+df["Total eve calls"]+df["Total night calls"]+df["Total intl calls"]

    df["Total_minutes"] = df["Total day minutes"]+df["Total eve minutes"]+df["Total night minutes"]+df["Total intl minutes"]

    df["Service_pressure"] = df['Customer service calls']/(df["Account length"]+1)

    df["Call_Per_Day"] = df["Total_Calls"]/(df["Account length"]+1)

    df["min_per_day"] = df["Total_minutes"]/(df["Account length"]+1)

    return df

def save_features_data(df):
    Feature_eng_Data.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Feature_eng_Data, index=False)

def main():
    df = load_data()
    df = feature_engineering(df)
    save_features_data(df)
    print("Feature engineering completed successfully")

if __name__ == "__main__":
    main()
