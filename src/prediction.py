import pandas as pd
import joblib

from config import Test_Data, Model_Path, PREDICTIONS_PATH
from preprocess import data_cleaning
from feauture_engineering import feature_engineering

def load_data():
    df = pd.read_csv(Test_Data)
    return df

def add_customer_Id(df):
    df = df.copy()
    df["Customer_ID"] = range(1, len(df)+1)
    return df

def preprocessing(df):
    df = data_cleaning(df)
    df = feature_engineering(df)
    return df

def load_model():
    model = joblib.load(Model_Path)
    return model

def predict(df):
    model = load_model()
    X = df.drop("Customer_ID", axis=1)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    result = pd.DataFrame({"Customer_ID": df["Customer_ID"], "Prediction" : preds, "Churn_Probability" : probs[:, 1]})
    return result

def save_predictions(df):
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_PATH, index=False)
    print("Predictions Saved")

def main():
    df = load_data()
    df = add_customer_Id(df)
    df = preprocessing(df)
    prediction_df = predict(df)
    print(prediction_df.head())
    save_predictions(prediction_df)

if __name__=="__main__":
    main()